#!/usr/bin/env bash
# 对 output/ 下各实验目录中的 da_result.csv 批量生成附录分场景可视化（默认 ylim 见 run_experiment_viz.py）
# 排除 output/_archive。若需包含归档，自行改 find 条件。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

while IFS= read -r csv; do
  dir=$(dirname "$csv")
  name=$(basename "$dir")
  echo "=== $csv ($name) ==="
  rm -rf "$dir/plots"
  python run_experiment_viz.py "$csv" --label "$name" --out-dir "$dir/plots"
done < <(find "$ROOT/output" -path '*/_archive/*' -prune -o -name 'da_result.csv' -print | sort)

echo "Done."
