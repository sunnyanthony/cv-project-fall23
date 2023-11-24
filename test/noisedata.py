import torch
from torch.utils.data import Dataset

class NoiseDataset(Dataset):
    def __init__(self, size, length):
        """
        size: noise size
        length: data length
        """
        self.size = size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.randn(self.size)