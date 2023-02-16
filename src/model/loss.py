import torch.nn as nn
import torch

def bce_loss(output, target):
    loss_func = nn.BCELoss()
    loss = loss_func(output, target)
    return loss