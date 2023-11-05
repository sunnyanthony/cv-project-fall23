from torch.utils.data import DataLoader
from torch import nn, ones_like, zeros_like, optim


class GAN():
    def __init__(self,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 groundtruth: DataLoader,
                 generator_input: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 learning_rate=1e-3) -> None:
        """
        Initialize the Generative Adversarial Network (GAN) with the 
        specified generator and discriminator networks, along with their
        corresponding data loaders.

        The GAN class aims to train the generator network to produce data
        that is indistinguishable from real data, as determined by the
        discriminator network.

        Parameters:
        generator (nn.Module): A neural network model that serves as the GAN's generator,
                which tries to generate data that appears to be from the
                same distribution as the ground truth data.
        discriminator (nn.Module): A neural network model that serves as the GAN's discriminator
                which tries to distinguish between genuine data and fake
                data produced by the generator.
        groundtruth (DataLoader): DataLoader that provides batches of real data against which
                the discriminator is trained to compare the generator's output.
        generator_input (DataLoader): DataLoader that provides batches of input noise vectors or
                seed data for the generator to produce its fake data.
        criterion (nn.Module): The loss function used for training the GAN. Common choices 
                include nn.BCELoss for binary classification tasks.
        optimizer (optim.Optimizer): The type of optimizer used to apply gradient updates 
                during training. Examples include optim.Adam or optim.SGD.
        learning_rate (float): The learning rate for the optimizers.

        Returns:
        None
        """
        self.gen = generator
        self.disc = discriminator
        self.gen_input = iter(generator_input)
        self.disc_input = iter(groundtruth)
        self.criterion = criterion
        self.gen_opt = optimizer(self.gen.parameters(), lr = learning_rate)
        self.disc_opt = optimizer(self.disc.parameters(), lr = learning_rate)

    @staticmethod
    def loss_calculation(true_loss, false_loss, proportion):
        """
        Calculate the discriminator's combined loss by taking a weighted average of the 
        true and false loss, where the true loss is the error on real data and the false 
        loss is the error on generated (fake) data.
        
        Parameters:
        true_loss: The discriminator's loss on real data.
        false_loss: The discriminator's loss on generated data.
        proportion: The weight for the true loss in the combined calculation.
        
        Returns:
        A weighted average of the true and false loss.
        """
        return (proportion * true_loss + (1 - proportion) *false_loss)

    def generator_loss(self, inputs):
        """
        Compute the loss for the generator, which measures how well the generator is 
        fooling the discriminator. The generator aims to produce outputs that the 
        discriminator classifies as real.
        
        Returns:
        The loss value for the generator.
        """

        outs = self.gen(inputs)
        checked = self.disc(outs)
        return self.criterion(checked, ones_like(checked))

    def discriminator_loss(self):
        """
        Compute the loss for the discriminator, which measures how well the discriminator 
        distinguishes between real and generated images. The discriminator's loss is a 
        combination of how well it recognizes real images as real (should_be_true) and 
        fake images as fake (should_be_false).
        
        Returns:
        The loss value for the discriminator.
        """

        inputs = next(self.gen_input)
        truth_inputs = next(self.disc_input)
        outs = self.gen(inputs)
        checked = self.disc(outs.detach())
        results = self.disc(truth_inputs)
        should_be_true = self.criterion(results, ones_like(results))
        should_be_false = self.criterion(checked, zeros_like(checked))
        proportion = results.shape[0] / (results.shape[0] + checked.shape[0])
        return self.loss_calculation(should_be_true, should_be_false, proportion), inputs

    def training(self):
        """
        Perform one iteration of training for both the discriminator and the generator.
        First, the discriminator's weights are updated by computing its loss and 
        applying backpropagation. The generator's weights are then updated based on 
        its loss, which is calculated by how well it fools the discriminator.
        """

        self.disc_opt.zero_grad()
        disc_loss, gen_inputs = self.discriminator_loss()
        disc_loss.backward(retain_graph=True)
        self.disc_opt.step()

        self.gen_opt.zero_grad()
        gen_loss = self.generator_loss(gen_inputs)
        gen_loss.backward()
        self.gen_opt.step()