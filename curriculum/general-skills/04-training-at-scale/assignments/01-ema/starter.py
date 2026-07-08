import torch


class EMA:
    """参数的指数滑动平均：shadow = decay*shadow + (1-decay)*param。"""
    def __init__(self, model, decay=0.99):
        raise NotImplementedError

    @torch.no_grad()
    def update(self, model):
        """更新影子参数（指数滑动平均）。"""
        raise NotImplementedError

    @torch.no_grad()
    def copy_to(self, model):
        """将影子参数复制回模型参数。"""
        raise NotImplementedError
