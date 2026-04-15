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


git config --global user.email sathya.pranav.deepak@gmail.com
git config --global user.name PranavDeepakSathya
