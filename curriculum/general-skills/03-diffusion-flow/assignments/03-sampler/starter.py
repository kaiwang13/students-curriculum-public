import torch


def euler_step(x, v, dt):
    """ODE 欧拉积分一步：x_{t+dt} = x + v * dt。"""
    raise NotImplementedError


def integrate(x0, velocity_fn, n_steps):
    """从 x0 出发，用常速度场 velocity_fn(x,t) 在 [0,1] 上欧拉积分 n_steps 步。"""
    raise NotImplementedError
