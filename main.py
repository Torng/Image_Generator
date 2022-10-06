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
                               # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                         shuffle=True)

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


# For each epoch
def cal_gradient_penalty(D, real, fake):
    # 每一个样本对应一个sigma。样本个数为64，特征数为512：[64,512]
    sigma = torch.rand(real.size(0), 1, device=device)  # [64,1]
    sigma = sigma.expand(real.size())  # [64, 512]
    # 按公式计算x_hat
    x_hat = sigma * real + (torch.tensor(1., device=device) - sigma) * fake
    x_hat.requires_grad = True
    # x_hat.to(device)
    # 为得到梯度先计算y
    d_x_hat = D(x_hat).to(device)

    # 计算梯度,autograd.grad返回的是一个元组(梯度值，)
    gradients = torch.autograd.grad(outputs=d_x_hat, inputs=x_hat,
                                    grad_outputs=torch.ones(d_x_hat.size(), device=device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    # 利用梯度计算出gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
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
        loss_D = -(lossr - lossf)  # max
        gradient_penalty = cal_gradient_penalty(netD, real_cpu, xf)
        loss_D = loss_D + gradient_penalty * 0.5
        optimizerD.zero_grad()
        loss_D.backward()
        optimizerD.step()
    z = torch.randn(b_size, nz, 1, 1, device=device)
    xf = netG(z)
    predf = netD(xf)
    loss_G = predf.mean()  # min
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
