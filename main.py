import torchvision

from Module.discriminator import Discriminator
from Module.generator import Generator
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from pathlib import Path
from torchvision.utils import save_image

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 500

# Learning rate for optimizers
lr = 0.00005

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

dataset = dset.ImageFolder(root="celeb_data",
                           transform=transforms.Compose([
                               transforms.Resize((64, 64)),
                               transforms.CenterCrop((64, 64)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                         shuffle=True, drop_last=True)

# Decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
netD = Discriminator().to(device)
netG = Generator().to(device)
# Setup Adam optimizers for both G and D
optimizerG = optim.RMSprop(netG.parameters(), lr=lr)
optimizerD = optim.RMSprop(netD.parameters(), lr=lr)

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")


def gradient_penalty(critic, real_image, fake_image, device="cpu"):
    batch_size, channel, height, width = real_image.shape
    # alpha is selected randomly between 0 and 1
    alpha = torch.rand(batch_size, 1, 1, 1, device=device).repeat(1, channel, height, width)
    # interpolated image=randomly weighted average between a real and fake image
    # interpolated image ← alpha *real image  + (1 − alpha) fake image
    interpolatted_image = (alpha * real_image) + (1 - alpha) * fake_image
    interpolatted_image = interpolatted_image.to(device)
    # calculate the critic score on the interpolated image
    interpolated_score = critic(interpolatted_image)

    # take the gradient of the score wrt to the interpolated image
    gradient = torch.autograd.grad(inputs=interpolatted_image,
                                   outputs=interpolated_score,
                                   retain_graph=True,
                                   create_graph=True,
                                   grad_outputs=torch.ones_like(interpolated_score, device=device)
                                   )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


iterator = iter(dataloader)
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for _ in range(5):  # train Dnet 5 times
        data = next(iterator)
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        predr = netD(real_cpu).view(-1)
        # maximize predr, therefore minus sign
        lossr = predr.mean()
        z = torch.randn(b_size, nz, 1, 1, device=device)
        xf = netG(z).detach()  # gradient would be passed down
        predf = netD(xf)
        # min predf
        lossf = predf.mean()
        loss_D = lossf - lossr  # max
        gradient_penalty = gradient_penalty(netD, real_cpu, xf, device)
        loss_D = loss_D + gradient_penalty * 10
        optimizerD.zero_grad()
        loss_D.backward()
        optimizerD.step()
    z = torch.randn(b_size, nz, 1, 1, device=device)
    xf = netG(z)
    predf = netD(xf)
    loss_G = -predf.mean()  # min
    # optimize
    optimizerG.zero_grad()
    loss_G.backward()
    optimizerG.step()

    if epoch % 2 == 0:
        print("epoch:{0} ==> lossDr:{1}, lossDf:{2}, lossD:{3},lossG:{4}".format(epoch, lossr, lossf, -loss_D, loss_G))
    if epoch % 10 == 0:
        path = Path("model_set/")
        path.mkdir(exist_ok=True)
        output_path = path / ("model_" + str(epoch))
        torch.save(netG, output_path)
