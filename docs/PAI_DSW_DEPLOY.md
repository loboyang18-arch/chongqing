# 阿里云 PAI-DSW 上运行 V16d Transformer

## 1. 本地：推代码到 GitHub（私有仓库）

```bash
cd chongqing_prj
git init
git add .
git commit -m "V16d Transformer + CUDA / PAI-DSW 部署"
# 在 GitHub 新建私有仓库后：
git remote add origin https://github.com/<你的用户>/<仓库名>.git
git branch -M main
git push -u origin main
```

> **说明**：推送需本机已配置 GitHub 凭据（HTTPS Personal Access Token 或 SSH key）。仓库名示例：`chongqing_prj`。

## 2. 本地：打包数据（不入库）

```bash
./scripts/package_v16d_data.sh
```

生成 `chongqing_v16d_data.tar.gz`，用 JupyterLab 上传、OSS 或 `wget` 传到 DSW。

## 3. DSW：克隆与解压

```bash
git clone https://github.com/<你的用户>/<仓库名>.git chongqing_prj
cd chongqing_prj
# 将 chongqing_v16d_data.tar.gz 放到当前目录后：
bash setup_server.sh
```

## 4. DSW：训练

```bash
bash run_gpu.sh
```

或：

```bash
python run_v16d_large_engfeat_noprice.py
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `V16D_DL_NUM_WORKERS` | DataLoader `num_workers`，默认 CUDA 为 4，CPU/MPS 为 0 |

## GPU 与 PyTorch

DSW 镜像通常已带 CUDA 版 PyTorch。在终端执行：

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
```

若 `cuda.is_available()` 为 False，按镜像文档安装与 CUDA 匹配的 `torch`。
