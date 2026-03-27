#!/usr/bin/env bash
# V16d + 工程特征（去价格列）训练，使用 CUDA（若可用）
# 可选覆盖:
#   V16D_EPOCHS=500 V16D_BS=64 V16D_D_MODEL=192 V16D_DIM_FF=576
#   V16D_N_LAYERS=4 V16D_N_HEAD=8 V16D_DROPOUT=0.2
#   V16D_OUT_DIR=v16d_probe_xl
set -euo pipefail
cd "$(dirname "$0")"

python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu/mps')"

exec python run_v16d_large_engfeat_noprice.py
