from typing import Any
from torch.utils.data import DataLoader
from torch import nn, optim, autograd, device, cuda, backends
from torch import ones_like, zeros_like, mean, rand, ones


class wgan_criterion(nn.Module):
    def __init__(self, disc, lambda_gp) -> None:
        self.lambda_gp = lambda_gp
        self.disc = disc
        self.device = device("cuda" if cuda.is_available() else "cpu" if not backends.mps.is_available() else "mps")
        #super().__init__(*args, **kwargs)
    
    def __call__(self, fake, real=None) -> Any:
        return mean(fake) - (mean(real) if real else 0) + \
            self.compute_gradient_penalty(self.disc, real, fake, self.device)

    def compute_gradient_penalty(self, Disc, real_data, fake_data, device):
        alpha = rand(real_data.size(0), 1, 1, 1, requires_grad=True).to(device)
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).to(device)
        interpolates = interpolates.requires_grad_(True)
        d_interpolates = Disc(interpolates)
        fake = ones(d_interpolates.size()).requires_grad_(False).to(device)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        return gradient_penalty

class stylegan_criterion(nn.Module):
    def __init__(self, disc, lambda_gp) -> None:
        self.lambda_gp = lambda_gp
        self.disc = disc
        self.device = device("cuda" if cuda.is_available() else "cpu" if not backends.mps.is_available() else "mps")
        #super().__init__(*args, **kwargs)
    
    def __call__(self, fake, real=None) -> Any:
        return self.compute_stylegan_discriminator_loss(self.disc, real, fake, self.device)

    def compute_stylegan_discriminator_loss(discriminator, real_images, fake_images, gamma=10.0):
        # Hinge loss for real images
        hinge_real = -torch.mean(torch.min(0, -1 + discriminator(real_images)))

        # Hinge loss for fake images
        hinge_fake = -torch.mean(torch.min(0, -1 - discriminator(fake_images)))

        # R1 regularization
        real_images.requires_grad = True
        logits_real = discriminator(real_images)
        gradients = torch.autograd.grad(outputs=logits_real.sum(), inputs=real_images, create_graph=True)[0]
        r1_penalty = 0.5 * gamma * torch.mean(gradients.pow(2))

        # Total discriminator loss
        discriminator_loss = hinge_real + hinge_fake + r1_penalty

        return discriminator_loss

class BEGANDiscriminatorLoss(torch.nn.Module):
    def __init__(self, lambda_=0.001, gamma=0.75):
        super(BEGANDiscriminatorLoss, self).__init__()
        self.lambda_ = lambda_
        self.gamma = gamma
        self.k_t = torch.tensor(0.0, requires_grad=False)

    def forward(self, real_images, fake_images, discriminator, generator):
        # Autoencoder reconstruction loss for real images
        recon_real = F.l1_loss(real_images, discriminator(real_images))

        # Autoencoder reconstruction loss for fake images
        fake_images_gen = generator(torch.randn_like(fake_images))
        recon_fake = F.l1_loss(fake_images, discriminator(fake_images_gen.detach()))

        # Convergence measure
        balance = F.l1_loss(discriminator(fake_images_gen) - self.lambda_ * fake_images, torch.zeros_like(fake_images))

        # Total discriminator loss
        loss_d = recon_real - self.k_t * balance
        return loss_d

losses = {
    "tranditional": nn.BCELoss, # -y log x - (1-y) log (1-x) => probabilities
    "wgan": wgan_criterion, # -y x + (1-y)(1-x) => score
    "stylegan": stylegan_criterion,
    "BEGAN": BEGANDiscriminatorLoss,
}


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
    def loss_calculation(true_loss, false_loss, proportion) -> int:
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
        if inputs is None or truth_inputs is None:
            return None, None
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

        while True:
            self.disc_opt.zero_grad()
            disc_loss, gen_inputs = self.discriminator_loss()
            if disc_loss is None or gen_inputs is None:
                # devour all data from dataloader
                break
            disc_loss.backward(retain_graph=True)
            self.disc_opt.step()

            self.gen_opt.zero_grad()
            gen_loss = self.generator_loss(gen_inputs)
            gen_loss.backward()
            self.gen_opt.step()
    
    def training_loop(self, epoch):
        for _ in range(epoch):
            self.training()
