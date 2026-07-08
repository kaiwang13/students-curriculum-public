# W9 平台作业（扩散模型与流匹配 · GA3）

| 平台 | 内容 | 目的 | 能力 |
|------|------|------|------|
| Lilian Weng "What are Diffusion Models?" (2021) <https://lilianweng.github.io/posts/2021-07-11-diffusion-models/> | 系统性综述：从 DDPM 前向/反向过程、分数匹配（score matching）、DDIM 确定性采样，到扩散模型的统一 SDE/ODE 框架；配有大量推导与图示 | 建立扩散模型全局视图；理解 $\bar{\alpha}_t$ 的几何意义与 predict-$\epsilon$ 目标；为 `w9-ddpm` 题包和 `01-ddpm` 课程笔记提供理论背景 | GA3 |
| Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020) <https://arxiv.org/abs/2006.11239> | DDPM 原始论文：推导前向加噪马尔科夫链的闭合形式 $q(x_t\|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$、predict-$\epsilon$ 简化目标函数、反向采样 Algorithm 1/2；线性噪声表的实验配置 | 掌握 DDPM 前向/反向过程的完整数学推导；理解噪声表对模型性能的影响；为 `w9-ddpm`（`linear_beta_schedule`/`q_sample`）题包提供一手文献依据 | GA3 |
| Lipman et al. "Flow Matching for Generative Modeling" (ICLR 2023) <https://arxiv.org/abs/2210.02747> | Flow Matching 原始论文：引入条件流匹配（CFM）目标；证明条件速度场期望等价于边际速度场（关键定理）；OT-CFM 用最优传输耦合构造线性插值路径 $x_t=(1-t)x_0+tx_1$，目标速度场 $v^*=x_1-x_0$ | 理解流匹配与扩散模型的本质区别（ODE vs SDE）；掌握 CFM 目标等价性证明；为 `w9-flow-matching`（`ot_interpolate`/`velocity_target`）题包和 `02-flow-matching` 课程笔记提供理论基础 | GA3 |

做法：先通过 Lilian Weng 综述建立全局认知，再精读 Ho et al. 掌握 DDPM 数学细节（对应 `w9-ddpm` 题包），最后阅读 Lipman et al. 理解流匹配框架（对应 `w9-flow-matching` 和 `w9-sampler` 题包）。