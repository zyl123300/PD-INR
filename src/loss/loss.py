import torch
import torch.nn as nn

def calc_mse_loss(loss, x, y):
    """
    Calculate mse loss.
    """
    # Compute loss
    loss_mse = torch.mean((x-y)**2)
    loss["loss"] += 1.0 * loss_mse
    loss["loss_mse"] = loss_mse
    return loss

def calc_huber_loss(loss, x, y, delta = 0.001):
    loss_huber = nn.functional.huber_loss(y, x, delta = delta)
    loss["loss"] += loss_huber
    loss["loss_huber"] = loss_huber
    return loss

def calc_gmc_loss(loss, x, y, c = 0.001):
    z = (y - x) ** 2
    loss_gmc = torch.mean(z / (1 + z / (c ** 2)))
    loss["loss"] += loss_gmc
    loss["loss_huber"] = loss_gmc
    return loss

def calc_tv_loss(loss, x, lam = 1e-5):
    """
    Calculate total variation loss.
    Args:
        x (n1, n2, n3, 1): 3d density field.
        k: relative weight
    """
    n1, n2, n3, _ = x.shape
    tv_1 = torch.abs(x[1:,1:,1:]-x[:-1,1:,1:]).sum()
    tv_2 = torch.abs(x[1:,1:,1:]-x[1:,:-1,1:]).sum()
    tv_3 = torch.abs(x[1:,1:,1:]-x[1:,1:,:-1]).sum()
    tv = (tv_1+tv_2+tv_3) / (n1*n2*n3)
    loss["loss"] += tv * lam
    loss["loss_tv"] = tv * lam
    return loss

def calc_poisson_loss(loss, x, y, lam = 1e-3):
    poisson = nn.PoissonNLLLoss(log_input = False)
    # mask_positive = x > 0
    # out = poisson(x[mask_positive], y[mask_positive])
    out = poisson(x, y)
    loss["loss"] += out * lam
    loss["loss_poisson"] = out * lam
    return loss
def calc_TV_loss(loss,tv_gradient,gamma=1):
    tv = torch.sum(tv_gradient,dim = -1,keepdim = False)
    # tv = tv / tv.shape[0]
    # tv = torch.sum(tv, dim = -1,keepdim = False)
    loss["tv_gradient"] = tv * gamma
    return loss
def calc_TV2_loss(loss,tv_gradient2,gamma=1):
    tv = torch.sum(tv_gradient2,dim = -1,keepdim = False)
    # tv = tv / tv.shape[0]
    # tv = torch.sum(tv, dim = -1,keepdim = False)
    loss["tv_gradient2"] = tv * gamma
    return loss