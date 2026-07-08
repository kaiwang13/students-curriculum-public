"""W6 VAE 阶段关卡：从零实现变分自编码器。见 projects/p-w6-vae/README.md。

架构说明：
  编码器（Encoder）：输入 x → 隐层 → (mu, logvar)，输出潜变量分布参数。
  重参数化（Reparameterize）：z = mu + eps * std，eps ~ N(0,1)，使梯度可回传。
  解码器（Decoder）：z → 重构 x_hat。
  损失函数 ELBO = 重构损失（MSE）+ β * KL 散度（KL = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)）。
"""
import torch
import torch.nn as nn


class VAE(nn.Module):
    """变分自编码器。

    Args:
        n_in (int): 输入维度。
        n_latent (int): 潜空间维度。
    """

    def __init__(self, n_in, n_latent):
        super().__init__()
        # TODO: 定义编码器（enc）、均值层（mu）、对数方差层（logvar）、解码器（dec）
        raise NotImplementedError

    def encode(self, x):
        """前向编码：x → (mu, logvar)。"""
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        """重参数化：z = mu + eps * std，std = exp(0.5 * logvar)。"""
        raise NotImplementedError

    def forward(self, x):
        """完整前向传播：x → (x_hat, mu, logvar)。"""
        raise NotImplementedError


def build_vae(n_in, n_latent):
    """构建并返回 VAE 模型实例。"""
    raise NotImplementedError


def elbo_loss(x_hat, x, mu, logvar, beta=1.0):
    """计算 ELBO 损失：recon_loss + beta * kl_loss。

    recon_loss = mean over samples of sum((x_hat - x)^2 over features)
    kl_loss    = mean over samples of 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
    """
    raise NotImplementedError


def train_vae(X, epochs=300, lr=1e-2, seed=0):
    """在数据矩阵 X (torch.Tensor 或可转张量) 上训练 VAE，返回 (model, final_loss)。

    使用 Adam 优化器，固定随机种子以保证可复现。
    """
    raise NotImplementedError
