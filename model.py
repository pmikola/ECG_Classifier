import torch
from mpmath import sigmoid
from torch import nn


class EcgClassifier(nn.Module):
    def __init__(self, no_labels: int):
        super().__init__()
        self.layer_0 = nn.Linear(187,100)
        self.layer_1 = nn.Linear(100, 50)
        self.layer_2 = nn.Linear(50, 10)
        self.layer_3 = nn.Linear(10, no_labels)

    def forward(self,x:torch.Tensor):
        x = self.layer_0(x)
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

