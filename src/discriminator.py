from torch import nn


class DESC(nn.Module):
    def __init__(self,
                 input_size):
        super().__init__()
    
    def forward(self, x):
        ...