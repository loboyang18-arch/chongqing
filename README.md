# 重庆电力现货市场 — 电价预测工程

基于公开运营数据，构建**日前（DA）**与**实时（RT）**小时级电价预测；主线为 **Level + Shape** 与 **V5 Profile**（日级均价、振幅缩放、RT spike 分流）。详细结论见根目录 **`实验报告.md`**，汇报速读见 **`实验报告_精简版.md`**。

---

## 目录结构

| 路径 | 说明 |
|------|------|
| `source_data/` | **必需** — 27 个原始 CSV（勿提交敏感未公开数据时自行准备） |
| `src/` | 数据流水线、特征工程、各阶段模型与评估脚本 |
| `params/` | **纳入版本库** — LightGBM 调参得到的 `tuning_da_best_params.json` / `tuning_rt_best_params.json` |
| `output/` | **仅运行时生成** — DWS 宽表、特征表、预测结果、图等（目录已保留，内容默认不提交 Git） |
| `notebooks/` | 探索与评估笔记本（可选） |
| `数据入库方案.md` | ODS→DWD→DWS 入库说明（中文） |
| `*.docx`（根目录） | 库表设计、字段规范等 Word 资料，**不参与** `python -m src.*` 运行 |

---

## 环境

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

建议在项目根目录执行，以便 `python -m src.xxx` 能解析包路径。

---

## 推荐运行顺序（从零生成可训练数据与当前主线模型）

### 1. 数据入库：ODS → DWD → DWS

```bash
python -m src.pipeline              # 完整流程（含节点电价，较慢）
# 或调试加速：
python -m src.pipeline --skip-nodal
```

产出（在 `output/`）：

- `dws_hourly_features.csv`（及可选 `.parquet`）
- `quality_report.csv`

### 2. 特征工程：小时宽表 → DA/RT 训练表

```bash
python -m src.feature_engineering
```

产出：

- `output/feature_da.csv`
- `output/feature_rt.csv`（含 V7 用的 **同类日模板** 列 `template_dev_h*`）

### 3. 当前推荐模型：V5 Profile

依赖：`params/tuning_*_best_params.json`（仓库已带）、上一步的 `feature_*.csv`。

```bash
python -m src.model_v5_profile
```

产出：`output/v5_profile/`（`summary.csv`、`da_result.csv`、`rt_result.csv`、叠图等）。

### 4. 其他脚本（按需）

| 模块 | 命令 | 说明 |
|------|------|------|
| 基线 LGB | `python -m src.model_baseline` | 早期对照 |
| Optuna 调参 | `python -m src.model_tuning` | 会覆盖写入 `params/tuning_*_best_params.json` |
| V4 Shape | `python -m src.model_v4_shape` | Huber + Level+Shape 重训练 |
| V3 增量实验 | `python -m src.model_v3_optimize` | spread/分位数/消融等 |
| V6 DA/RT 分线 | `python -c "from src.model_v5_profile import run_v6_all; run_v6_all()"` | Level 自适应 / RT shape gating |
| **V7 结构化** | `python -m src.model_v7_structural` | 模板+残差、风险分、扩展 shape 指标；产出 `output/v7_*` |
| 形状横评 | `python -m src.evaluate_shapes` | 需先有各 `*_result.csv` |

---

## 设计要点（极简）

- **防泄漏**：特征按 `lag0 / lag1(24h) / lag2(48h)` 分层；状态类阈值用 **expanding** 分位等避免偷看未来。
- **评估**：除 MAE/RMSE 外，使用 `src/shape_metrics.py` 的 **profile 相关、方向准确率、振幅误差** 等；V5 含 **Composite Score** 选模；V7 可选 **转折点 / 分块** 指标（`compute_shape_report(..., include_v7=True)`）。
- **超参 JSON**：已从 `output/` 迁至 **`params/`**，避免与一次性运行结果混在一起。

---

## 文档

- **`实验报告_精简版.md`** — 结论与数字，适合汇报  
- **`实验报告.md`** — 完整技术版  
- **`数据入库方案.md`** — 入库字段与流程补充  

---

## 许可与数据

原始数据版权归提供方；本仓库代码仅供研究/内部使用，请遵守数据来源与保密要求。
