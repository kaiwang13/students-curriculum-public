# W6 VAE 阶段关卡 `p-w6-vae`

对应能力 Competencies: **SH9**（变分自编码器与生成建模）
通过线 Pass threshold: 自动评分 ≥ 0.7，量表每维度 ≥ 3/4

---

## 任务说明

从零实现一个变分自编码器（VAE），并在合成表格数据上训练它。

| 步骤 | 内容 |
|------|------|
| 1. 生成数据 | 调用 `make_synth_table` 生成合成表格数值型特征，标准化后转为 `torch.Tensor` |
| 2. 实现 VAE | 完成 `VAE` 类：`encode → reparameterize → decode` |
| 3. 实现损失 | `elbo_loss = 重构损失（MSE）+ β × KL 散度` |
| 4. 训练 | `train_vae(X, epochs=300)` 返回 `(model, final_loss)`；训练必须使 loss 下降 |
| 5. 可视化 | 绘制重构 vs 原始对比图（至少 1 张，带轴标签和标题） |

### 接口规范

```python
class VAE(nn.Module):
    def __init__(self, n_in: int, n_latent: int): ...
    def encode(self, x) -> tuple[Tensor, Tensor]: ...       # (mu, logvar)
    def reparameterize(self, mu, logvar) -> Tensor: ...      # z
    def forward(self, x) -> tuple[Tensor, Tensor, Tensor]:  # (x_hat, mu, logvar)

def build_vae(n_in: int, n_latent: int) -> VAE: ...

def elbo_loss(x_hat, x, mu, logvar, beta=1.0) -> Tensor: ...

def train_vae(X, epochs=300, lr=1e-2, seed=0) -> tuple[VAE, float]: ...
```

**ELBO 公式**：

```
recon = mean_over_samples( sum_over_features( (x_hat - x)^2 ) )
kl    = mean_over_samples( 0.5 * sum_over_latent( exp(logvar) + mu^2 - 1 - logvar ) )
loss  = recon + beta * kl
```

---

## 项目结构

```
progress/<你的名字>/
└── work/
    └── p-w6-vae/
        ├── solution.py        ← 实现 VAE / build_vae / elbo_loss / train_vae（必须）
        └── analysis.ipynb     ← 训练过程 + 重构 vs 原始对比图
```

---

## 自动评分

```bash
# 从仓库根目录运行（使用 pt29 环境）
python check.py p-w6-vae
```

通过条件：`score >= 0.7`（即 3 项测试中至少通过 2 项）。

评分项目：
1. `test_forward_shapes` — 输出形状正确（`x_hat` 与输入同形，`mu`/`logvar` 为潜变量维度）
2. `test_training_reduces_loss` — 训练 300 步后 ELBO loss 低于初始 loss
3. `test_latent_dim` — `encode` 输出的 `mu` 维度 == `n_latent`

---

## 评分量表（导师评分）

详见 `assessments/rubrics/p-w6-vae.md`

维度：正确性 / 代码质量 / 评估严谨性 / 表达

---

## 参考资料

- W6 Lesson 04：`lessons/04-vae.ipynb`
- W6 练习：`w6-vae`（`reparameterize` + `kl_divergence`）
- 平台：`assessments/platforms/w6.md`（Karpathy VAE 视频 + PyTorch 教程）
- Kingma & Welling (2013) "Auto-Encoding Variational Bayes"