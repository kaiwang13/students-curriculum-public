# W7 平台作业（Transformer 架构 · GA1）

| 平台 | 内容 | 目的 | 能力 |
|------|------|------|------|
| Karpathy "Let's build GPT" (YouTube) | 从零手写 GPT-2（nanoGPT）：多头注意力、因果掩码、Transformer 块、训练循环；用 Python + PyTorch，约 1 小时讲座 | 建立注意力机制与 Transformer 的直觉；理解因果掩码在自回归生成中的必要性；与 `w7-attention` 和 `w7-mha` 题包直接对应 | GA1 |
| The Annotated Transformer (Harvard NLP) | 带逐行注释的完整 Transformer PyTorch 实现：缩放点积注意力、多头注意力、位置编码、编码器-解码器架构 | 深化对 Transformer 各模块数学推导的理解；提供可运行参考代码，与课程笔记 `01-attention`/`02-multihead` 配套 | GA1 |
| Gorishniy et al. "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021) | FT-Transformer 原始论文：FeatureTokenizer 设计、[CLS] token 作为全局表征、与 ResNet 等 baseline 的对比实验 | 理解 FT-Transformer 将 Transformer 应用于表格数据的核心思路；为 `w7-ft-transformer` 题包和 `03-ft-transformer` 课程笔记提供理论基础 | GA1 |

做法：先通过 Karpathy 讲座建立 Transformer 直觉（对应 w7-attention/w7-mha 题包），再用 Annotated Transformer 深化多头注意力实现细节，最后阅读 FT-Transformer 论文完成表格数据部分（对应 w7-ft-transformer 题包）。