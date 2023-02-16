import torch
import torch.nn as nn
from collections import OrderedDict


class MilabModel(nn.Module):
    # __init__
    # first index of node_sizes is inputsize
    # last index of node_sizes is outputsize
    def __init__(self, node_sizes):
        super(MilabModel, self).__init__()

        hidden_size = len(node_sizes)
        mlp_stack = OrderedDict()
        for i in range(hidden_size - 1):

            mlp_stack["fc_layer_" + str(i)] = nn.Linear(
                node_sizes[i], node_sizes[i + 1]
            )
            mlp_stack["batch_norm_" + str(i)] = nn.BatchNorm1d(node_sizes[i + 1])
            mlp_stack["relu_" + str(i)] = nn.ReLU()
            mlp_stack["dropout_" + str(i)] = nn.Dropout()
        mlp_stack["fc_layer_last"] = nn.Linear(node_sizes[len(node_sizes) - 1], 2)

        self.model = nn.Sequential(mlp_stack)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    list = [5, 6, 5, 2]
    device = torch.device("cpu")
    model = MilabModel(list).to(device)

    x = torch.Tensor([0.0, 3.0, 4.0, 5.0, 1.0])
    y = model.forward(x)
