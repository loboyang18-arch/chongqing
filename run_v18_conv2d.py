#!/usr/bin/env python3
"""V18 Conv2D 入口脚本（默认预测日前出清电价 da_clearing_price，见 model_v18_conv2d.TARGET_COL）。

环境变量：
  V18_TARGET_COL  预测目标：列名如 da_clearing_price / rt_clearing_price；
                    或 spread_rt_minus_da（与 rt_da_spread 等价，小时 RT−DA 价差，见 run_v18_conv2d_spread.py）
  V18_EPOCHS   训练轮数  (默认 200)
  V18_BS       batch size (默认 64)
  V18_LR       学习率     (默认 1e-3)
  V18_OUT_DIR  输出子目录  (默认 v18_conv2d)
  V18_TRAIN_OVERSAMPLE   训练逻辑样本倍数，K>1 时每基础样本 K-1 个带标签残差扰动的副本 (默认 1=关)
  V18_OVERSAMPLE_RESID_SCALE  上述副本标签残差扰动强度 (默认 0.28)
  V18_RESIDUAL_MC       首份样本残差 MC (默认 0=关；1 开)
  V18_RESIDUAL_MC_P / SCALE / NPASS  仅作用于 rep=0 (默认 0.28 / 0.35 / 1)

在终端里希望日志尽量「边跑边打印」时，使用无缓冲（任选其一）：
  PYTHONUNBUFFERED=1 python run_v18_conv2d.py
  python -u run_v18_conv2d.py

推荐通过实验管道运行：
  python run_experiment.py --config experiments/v18_conv2d_default.yaml
"""
import logging
import os

from src.config import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

out_dir_name = os.environ.get("V18_OUT_DIR", "v18_conv2d")
out_dir = OUTPUT_DIR / out_dir_name
out_dir.mkdir(parents=True, exist_ok=True)

from src.model_v18_conv2d import run_v18

if __name__ == "__main__":
    run_v18(out_dir=out_dir)
