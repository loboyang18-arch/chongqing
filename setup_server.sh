#!/usr/bin/env bash
# PAI-DSW / Linux GPU 服务器：解压数据并安装 Python 依赖
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DATA_TAR="${1:-chongqing_v16d_data.tar.gz}"
if [[ -f "$DATA_TAR" ]]; then
  echo "解压数据: $DATA_TAR"
  tar xzf "$DATA_TAR" -C "$ROOT"
else
  echo "提示: 未找到 $DATA_TAR。请将数据包放到项目根目录，或: bash setup_server.sh /path/to/chongqing_v16d_data.tar.gz"
fi

echo "安装依赖（若已安装会较快）..."
python -m pip install -q --upgrade pip
python -m pip install -q "numpy>=1.24" "pandas>=1.5" "matplotlib>=3.7" "torch>=2.0"

echo "校验 GPU:"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "完成。运行: bash run_gpu.sh"
