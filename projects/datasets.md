# 数据集目录 Datasets

| 名称 | 来源 | 用途 | 加载 |
|------|------|------|------|
| 合成表格数据 (synthetic) | `data/make_synth_table.py` | 默认；离线可复现 | `make_synth_table(n, seed)` |
| 脏版合成数据 | `data/make_synth_table.py` | W3 数据管道项目 | `make_messy_table(n, seed)` |

原则：优先用合成数据（零下载）；所有练习均可离线完成。