#!/usr/bin/env bash
# V16d no-engfeat sweep (3 configs x multi-seed)
# 常用:
#   V16D_EPOCHS=300 V16_SWEEP_SEEDS=0,1,2 V16_SWEEP_OUT_DIR=v16d_noeng_sweep_e300
set -euo pipefail
cd "$(dirname "$0")"

python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu/mps')"

exec python run_v16d_noengfeat_sweep.py
