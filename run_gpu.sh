#!/usr/bin/env bash
# V16d-large + 工程特征（去价格列）训练，使用 CUDA（若可用）
# 可选：export V16D_EPOCHS=100 覆盖默认 500
set -euo pipefail
cd "$(dirname "$0")"

python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu/mps')"

exec python run_v16d_large_engfeat_noprice.py
