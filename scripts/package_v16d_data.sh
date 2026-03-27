#!/usr/bin/env bash
# 打包 V16d 训练所需数据（不含代码）。输出：项目根目录 chongqing_v16d_data.tar.gz
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="chongqing_v16d_data.tar.gz"

if [[ ! -d source_data ]]; then
  echo "错误: 缺少目录 source_data/"
  exit 1
fi

paths=(source_data)
[[ -f output/dws_hourly_features.csv ]] && paths+=(output/dws_hourly_features.csv) || echo "警告: 缺少 output/dws_hourly_features.csv（请先跑 pipeline 生成）"
[[ -f output/feature_da.csv ]] && paths+=(output/feature_da.csv) || echo "警告: 缺少 output/feature_da.csv（engfeat 脚本需要）"

echo "打包: ${paths[*]} -> $OUT"
tar czvf "$OUT" "${paths[@]}"
echo "完成: $(du -h "$OUT" | cut -f1) $OUT"
