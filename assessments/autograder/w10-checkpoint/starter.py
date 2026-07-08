from __future__ import annotations
from pathlib import Path
import torch


def save_checkpoint(model, optimizer, step, path):
    """保存模型/优化器状态与步数到 path。"""
    raise NotImplementedError


def load_checkpoint(model, optimizer, path):
    """从 path 恢复模型/优化器状态，返回 step。"""
    raise NotImplementedError
