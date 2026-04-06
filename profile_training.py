"""
Profile mdlARC training loop.

Usage:
    cd mdlARC
    python profile_training.py

Outputs:
    ./profiler_logs/  - TensorBoard trace (open with: tensorboard --logdir=profiler_logs)
    ./profile_trace.json - Chrome trace (open with: chrome://tracing)
    
    Also prints a summary table of top CUDA kernels sorted by total time.
"""

import sys
import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

SRC_DIR = Path.cwd() / "src"
sys.path.insert(0, str(SRC_DIR))

import utils
import train
import build

print("Modules imported.")

# Use low preset with minimal epochs - we only need a few steps to profile
args_dict = {
    "name": "profile_run",
    "data_path": Path("assets/challenges.json"),
    "train_log_file": None,
    "save_path": None,
    "checkpoint_path": None,
    "checkpoint_epochs": [],

    "epochs": 1,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "do_validate": False,
    "val_batch_size": 70,

    "enable_aug": True,
    "max_augments": 80,
    "enable_color_aug": True,
    "color_apply_to_test": True,
    "enable_dihedral_aug": True,
    "dihedral_apply_to_test": True,

    "optimizer": "normuon",
    "normuon_lr": 1.66e-3,
    "normuon_momentum": 0.95,
    "normuon_beta2": 0.95,
    "adamw_lr": 3e-4,

    "warmup_pct": 0.02,
    "wsd_decay_start_pct": 0.8,
    "lr_floor": 0.0,

    "weight_decay": 0.1,
    "attention_weight_decay": 0.01,
    "token_embedding_weight_decay": 0.01,
    "task_embedding_weight_decay": 0.01,

    "grad_clip": 1.0,
    "dropout": 0.1,
    "attention_dropout": None,
    "seed": 42,

    # Architecture - match his current config
    "d_model": 768,
    "n_heads": 12,
    "d_ff": 3072,
    "n_layers": 8,

    "inference_temperature": None,
    "inference_top_k": None,
    "train_log_mode": "never",
    "log_location": "none",
}

cfg = argparse.Namespace(**args_dict)
Path("runs").mkdir(parents=True, exist_ok=True)

# Build model and data
print("Building model and data...")
model, dataset, dataloader, device, data_path = build.build_model_and_data(cfg)

# Compile like he does in training
print("Compiling model...")
training_model = torch.compile(model)

# Setup optimizer (simplified - just AdamW for profiling)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Warmup steps - let torch.compile do its thing before we profile
print("Warming up (5 steps)...")
model.train()
warmup_iter = iter(dataloader)
for i in range(5):
    batch = next(warmup_iter)
    input_ids = batch["input_ids"].to(device)
    example_ids = batch["example_ids"].to(device)
    dihedral_ids = batch["dihedral_ids"].to(device)
    positions_3d = batch["positions_3d"].to(device)
    cu_seqlens = batch.get("cu_seqlens")
    max_seqlen = batch.get("max_seqlen")
    sep_indices = batch.get("sep_indices")

    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32)
        max_seqlen = int(max_seqlen.item()) if torch.is_tensor(max_seqlen) else int(max_seqlen)
        attention_mask = None
    else:
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

    if sep_indices is not None:
        sep_indices = sep_indices.to(device)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = training_model(
            input_ids, example_ids, dihedral_ids,
            attention_mask=attention_mask,
            sep_indices=sep_indices,
            compute_input_loss=False,
            positions_3d=positions_3d,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        loss = outputs["output_loss"]
    loss.backward()
    optimizer.step()
    print(f"  warmup step {i+1}/5 done, loss={loss.item():.4f}")

torch.cuda.synchronize()

# Now profile
print("\nProfiling (wait=1, warmup=2, active=5 steps)...")
prof_dir = Path("profiler_logs")
prof_dir.mkdir(exist_ok=True)

profile_iter = iter(dataloader)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=5, repeat=1),
    on_trace_ready=tensorboard_trace_handler(str(prof_dir)),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
) as prof:
    for i in range(8):  # wait(1) + warmup(2) + active(5)
        batch = next(profile_iter)
        input_ids = batch["input_ids"].to(device)
        example_ids = batch["example_ids"].to(device)
        dihedral_ids = batch["dihedral_ids"].to(device)
        positions_3d = batch["positions_3d"].to(device)
        cu_seqlens = batch.get("cu_seqlens")
        max_seqlen = batch.get("max_seqlen")
        sep_indices = batch.get("sep_indices")

        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32)
            max_seqlen = int(max_seqlen.item()) if torch.is_tensor(max_seqlen) else int(max_seqlen)
            attention_mask = None
        else:
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        if sep_indices is not None:
            sep_indices = sep_indices.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = training_model(
                input_ids, example_ids, dihedral_ids,
                attention_mask=attention_mask,
                sep_indices=sep_indices,
                compute_input_loss=False,
                positions_3d=positions_3d,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            loss = outputs["output_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        prof.step()
        print(f"  profile step {i+1}/8 done")

torch.cuda.synchronize()

# Print summary
print("\n" + "="*80)
print("TOP 30 CUDA KERNELS BY TOTAL CUDA TIME")
print("="*80)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

print("\n" + "="*80)
print("TOP 20 BY CUDA MEMORY USAGE")
print("="*80)
print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))

# Export chrome trace
trace_path = "profile_trace.json"
prof.export_chrome_trace(trace_path)
print(f"\nChrome trace exported to {trace_path}")
print(f"TensorBoard logs in {prof_dir}/")
print(f"\nTo view: tensorboard --logdir={prof_dir}")
print(f"Or open {trace_path} in chrome://tracing")