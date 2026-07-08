# 环境搭建 Setup

在 `student/` 根目录（本文件夹的上一级）执行。本课程用 [uv](https://docs.astral.sh/uv/) 管理依赖。

## 1. 安装 uv（一次性）
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Windows PowerShell：`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

## 2. 创建虚拟环境并安装依赖
```bash
uv venv --python 3.11
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r env/requirements.txt
```
依赖清单见 `env/requirements.txt`（含 torch / xgboost / statsmodels，CPU 版即可自测）。
> 只需 CPU 且想减小下载体积，可改装 CPU 版 torch：
> `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`

之后每开一个新终端，先 `source .venv/bin/activate` 再跑命令。

## 3. 验证环境
```bash
python check.py --list
```
能列出全部可检查的练习即表示就绪。`python check.py --progress` 看各章进度概览。

## 4. 怎么做一道练习
1. 进入某一章（如 `curriculum/foundations/02-numpy/`），先读 `lessons/`。
2. 在**同一章** `assignments/<练习>/starter.py` 里，把 `raise NotImplementedError` 换成你的实现。
3. 运行自测：
   ```bash
   python check.py <练习名>
   ```
   例：`python check.py 01-oop`。结果 ✅ 通过 / ❌ 未通过（附失败详情）/ ○ 未完成。

建议的完成顺序见根目录的 `ORDER.md`；各章说明见 `curriculum/`。