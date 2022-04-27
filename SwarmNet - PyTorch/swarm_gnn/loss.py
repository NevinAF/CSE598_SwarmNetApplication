from torch import nn


def retrieve_loss(name):
    if name.lower() == "mse":
        return nn.MSELoss(reduction='none')
