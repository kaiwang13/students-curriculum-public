# 环境搭建 Setup

在 `student/` 根目录（本文件夹的上一级）执行以下步骤。

## 1. 创建并激活 conda 环境
```bash
conda env create -f env/environment.yml && conda activate students
```
环境名为 `students`，已包含所有周所需依赖（含 torch / xgboost / statsmodels，CPU 版即可自测）。

## 2. 验证环境
```bash
python check.py --list
```
能列出全部可检查的题包，即表示环境就绪。也可以 `python check.py --progress` 看各周进度概览。

## 3. 怎么做一道练习
1. 编辑该题包的 `starter.py`，把里面的 `raise NotImplementedError` 换成你的实现：
   本章 `assignments/` 下对应练习的 `starter.py`
2. 运行自测：
   ```bash
   python check.py <题包名>
   ```
   例：`python check.py 01-oop`。结果 ✅ 通过 / ❌ 未通过（附失败详情）/ ○ 未完成。

建议的完成顺序见根目录的 `ORDER.md`；各周说明见 `curriculum/`。