from torch import nn
import torch

__all__ = [
    'Discriminator0',
    'Discriminator1',
]

def block(input_dim: int, output_dim: int, negative_slope=0.2):
    return nn.Sequential(*[
    nn.Linear(input_dim, output_dim),
    nn.LeakyReLU(negative_slope=negative_slope),
    ])

def cblock(input_dim: int, output_dim: int, negative_slope=0.2, convpara=(4, 2, 1)):
    return nn.Sequential(*[
            nn.Conv2d(input_dim, output_dim, *convpara, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
    ])

class Discriminator0(nn.Module):
    def __init__(self, img_dim: int = 1088*608, hidden_dim: int = 128):
        super().__init__()
        self.discriminator = nn.Sequential(*[
              block(img_dim, hidden_dim*4),
              block(hidden_dim*4, hidden_dim*2),
              block(hidden_dim*2, hidden_dim),
              nn.Linear(hidden_dim, 1),
            ])

    def forward(self, image: torch.Tensor):
        return self.discriminator(image)

class Discriminator1(nn.Module):
    updown_sampling = []
    def __init__(self, input: int = 1088*608):
        super().__init__()
        self.main = nn.Sequential(
            cblock(input, input * 2),
            cblock(input*2, input * 4),
            cblock(input*4, input * 8),
            nn.Conv2d(input * 8, 1, 4, 1, 0, bias=False),
            nn.Linear(input * 8, 1), # TBD: ...
        )

    def forward(self, input):
        return self.main(input)