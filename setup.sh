#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install ninja
pip install nvcc4jupyter numpy jupyter_client ipykernel ipywidgets
pip install triton
pip install torch
pip install matplotlib
pip install pandas
pip install nvidia-cutlass-dsl
pip install numba

# Flash attention (sm_120 only for 5090/CUDA 13)
git clone https://github.com/Dao-AILab/flash-attention.git flash-attention
cd flash-attention
FLASH_ATTN_CUDA_ARCHS="120" MAX_JOBS=16 pip install . --no-build-isolation
cd ..

# mdlARC
git clone https://github.com/mvakde/mdlARC.git mdlARC
cd mdlARC
pip install -r requirements.txt
cd ..

# Intra-kernel profiler
git clone https://github.com/yao-jz/intra-kernel-profiler.git /root/intra-kernel-profiler

git config --global user.email sathya.pranav.deepak@gmail.com
git config --global user.name PranavDeepakSathya
