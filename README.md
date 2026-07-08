# 研究工程通识课程 (Research-Engineering Foundations)

面向只学过入门 Python 的同学的自学课程，从工程与科学计算基础，一路到深度学习与统计方法的通识知识。
每个练习都配有可本地运行的自测。

## 从哪里开始
1. 按 `ORDER.md` 的顺序学习：**基础 → 通用技能 → 通用知识**。
2. 环境搭建见 `env/setup.md`；课程地图见 `curriculum/README.md`。
3. 做题：进入某一章（如 `curriculum/foundations/02-numpy/`），先读 `lessons/`，再到**同一章的**
   `assignments/<练习>/starter.py` 里把 `raise NotImplementedError` 换成你的实现，然后
   `python check.py <练习>` 自测（✅ 通过 / ❌ 未通过 / ○ 未完成）。

## 文件结构
```
├── README.md / ORDER.md          本说明 / 学习顺序清单
├── check.py                      自测入口：python check.py <练习名>
├── curriculum/
│   ├── foundations/              基础：Python · NumPy · pandas · 统计 · 机器学习 · 深度学习(含 nanograd L1–3)
│   ├── general-skills/           通用技能：张量自动微分 · Transformer · 扩散/流匹配 · 规模化训练 · 研究工程
│   └── general-knowledge/        通用知识：关联检验 · 群体结构与混合模型
│        每章 = lessons/(课程) + assignments/(练习, 在 starter.py 里实现) + README.md
├── assessments/                  quizzes 小测 · platforms 延伸阅读 · rubrics 评分量表
├── projects/data/                合成数据生成器（供练习使用）
└── env/                          环境搭建
```
先读某章 `lessons/*.ipynb`，再做该章 `assignments/` 里的练习。评分标准见 `assessments/rubrics/`。