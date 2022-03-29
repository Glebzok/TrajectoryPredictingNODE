import torch

def matrix_exp_odeint(rhs, y0, t):
    # y: bs x d
    # t: T
    # T, bs, d
    y = (torch.matrix_exp(t[:, None, None] * rhs.linear.weight[None, :, :]) @ y0.T).permute(0, 2, 1)
    # print(y0.shape, t.shape, y.shape)
    return y
