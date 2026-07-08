import torch


def accumulate_and_step(model, optimizer, microbatches, loss_fn):
    """梯度累积：对每个 microbatch 前向+反向累积梯度（损失按 microbatch 数归一化），
    最后 step 一次。返回累积的平均损失（float）。

    注意：此等价于全批次梯度仅对等大小的 microbatch 成立；若 microbatch 大小不等，
    应按样本数加权归一化。"""
    raise NotImplementedError
