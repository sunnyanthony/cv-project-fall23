import functools 
from src import gan
from test import test_models, noisedata, exampledata
from torch.utils.data import DataLoader
import torch.optim as optim

iter = 0
noise_size = 128

def show_generated_images(images, num_images=4):
    import matplotlib.pyplot as plt
    import numpy as np
    global iter

    iter += 1
    if iter % 100 != 0:
        return

    images = images.detach().cpu().numpy()
    fig, axes = plt.subplots(1, num_images, figsize=(10, 10))

    for i, ax in enumerate(axes):
        ax.imshow(np.squeeze(images[i]), cmap='gray')
        ax.axis('off')
    plt.show()

generator = test_models.Generator(noise_size)
disciminator = test_models.Discriminator()
disciminator_dataloader = DataLoader(exampledata.MNISTDataset("."), batch_size=32, shuffle=True)
generator_dataloader = DataLoader(noisedata.NoiseDataset(size=noise_size, length=len(disciminator_dataloader)*32), batch_size=32, shuffle=True)


g = gan.GAN(
    generator=generator,
    generator_input=generator_dataloader,
    discriminator=disciminator,
    groundtruth=disciminator_dataloader,
    criterion=gan.losses["tranditional"](),
    optimizer=functools.partial(optim.Adam, betas=(0.5, 0.999)),
    schefuler=functools.partial(optim.lr_scheduler.StepLR, step_size=30, gamma=0.1)
)

g.register_train_hook(show_generated_images)
g.training_loop(epoch=3)