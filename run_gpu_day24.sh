#!/usr/bin/env bash
# V17 日级 Transformer：1天样本 -> 24点输出
# 可选覆盖:
#   V17_EPOCHS=200 V17_BS=32 V17_D_MODEL=128 V17_DIM_FF=384
#   V17_N_LAYERS=3 V17_N_HEAD=4 V17_DROPOUT=0.2 V17_PAST_DAYS=14
#   V17_OUT_DIR=v17_day24_probe
set -euo pipefail
cd "$(dirname "$0")"

python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu/mps')"

exec python run_v17_day24_transformer.py
