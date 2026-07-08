# 算力路线图 Compute ramp（分阶段）

| 阶段 | 内容 | 算力 | 说明 |
|------|------|------|------|
| 基础 Foundations | Python → NumPy → pandas → 统计 → 机器学习 | 笔记本 CPU（可选 Colab） | 均可 CPU 小数据完成 |
| 深度学习入门 | Foundations 的深度学习 + nanograd L1–L3 + VAE | Colab GPU 或 单张实验室 GPU | 首个需要 GPU 的关卡（VAE） |
| 通用技能 General Skills | 张量自动微分 · Transformer · 扩散/流匹配 · 规模化训练 | 单张实验室 GPU | 扩散/流匹配训练相对较重 |
| 通用知识 General Knowledge | 关联检验 · 群体结构与混合模型 | 笔记本 CPU | 合成数据，CPU 即可 |

原则：能在 CPU 上做的练习就在 CPU 上做；GPU 只用于确实需要的训练。