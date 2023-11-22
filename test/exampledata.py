from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os

class MNISTDataset(Dataset):
    def __init__(self, root):
        """
        root: path to download
        """
        download = True if not self._check_exists(root, True) else False
        self.mnist = datasets.MNIST(root=root, train=True, download=download)
        self.transform = self._transform()

    def _transform(self):
        # put some augementation
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        # we don't need labels
        image, _ = self.mnist[idx]
        image = self.transform(image)
        return image

    def _check_exists(self, root, train):
        subfolder = 'MNIST/raw'
        files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"] if train else ["t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
        return all(os.path.exists(os.path.join(root, subfolder, file)) for file in files)
