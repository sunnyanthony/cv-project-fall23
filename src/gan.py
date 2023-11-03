from torch.utils.data import DataLoader
from torch import nn

class GAN():
    def __init__(self,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 groundtruth: DataLoader,
                 generator_input: DataLoader) -> None:
        """
        Initialize the Generative Adversarial Network (GAN) with the 
        specified generator and discriminator networks, along with their
        corresponding data loaders.

        The GAN class aims to train the generator network to produce data
        that is indistinguishable from real data, as determined by the
        discriminator network.

        Parameters:
        generator:       A neural network model that serves as the GAN's generator,
                         which tries to generate data that appears to be from the
                         same distribution as the ground truth data.
        discriminator:   A neural network model that serves as the GAN's discriminator
                         which tries to distinguish between genuine data and fake
                         data produced by the generator.
        groundtruth:     DataLoader that provides batches of real data against which
                         the discriminator is trained to compare the generator's output.
        generator_input: DataLoader that provides batches of input noise vectors or
                         seed data for the generator to produce its fake data.

        Returns:
        None
        """
        self.gen = generator
        self.disc = discriminator
        self.gen_input = generator_input
        self.disc_input = groundtruth