import torch


def matrix_exp_odeint(rhs, y0, t):
    """

    Parameters
    ----------
    rhs: nn.Module
        right-hand side of ODE to solve
    y0: torch.tensor
        (bs , dim) initial value
    t: torch.tensor
        (T, )
    Returns y
        (T, bs, dim) solution of initial value problem at times t
    -------

    """
    y = (torch.matrix_exp(t[:, None, None] * rhs.dynamics.weight[None, :, :]) @ y0.T).permute(0, 2, 1)
    return y
