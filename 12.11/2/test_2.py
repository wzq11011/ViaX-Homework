import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms as tfs
import torchvision.utils as vutils
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

tfs_portrait = tfs.Compose([
    tfs.CenterCrop(512),
    tfs.RandomHorizontalFlip(),
    tfs.Resize(64),
    tfs.ToTensor(),
    tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

tfs_background = tfs.Compose([
    tfs.CenterCrop(512),
    tfs.Resize(128),
    tfs.ToTensor(),
    tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

Portrait_noback_set = ImageFolder('./img_jpg', transform = tfs_portrait)
Portrait_noback_data = DataLoader(Portrait_noback_set, batch_size = 8, shuffle = True, drop_last = True)

Back_set = ImageFolder('./landmark1_big', transform = tfs_background)
Back_data = DataLoader(Back_set, batch_size = 8, shuffle = True, drop_last = True)

Portrait_withback_set = ImageFolder('./portrait_with_back', transform = tfs_portrait)
Portrait_withback_data = DataLoader(Portrait_withback_set, batch_size = 8, shuffle = True, drop_last = True)



class FCN_Background(nn.Module):
    def __init__(self):
        super(FCN_Background, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128))

        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256))

        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256))

        self.maxpool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512))

        self.maxpool4 = nn.MaxPool2d(2, 2)

        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(128, 3, 4, 2, 1)

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = self.maxpool3(self.conv3(x))
        x = self.maxpool4(self.conv4(x))
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64))
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128))

        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64))
        self.maxpool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32))
        self.maxpool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 1, 4, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = self.maxpool3(self.conv3(x))
        x = self.maxpool4(self.conv4(x))
        x = self.conv5(x)
        return x

back_FCN = FCN_Background()
back_FCN = nn.DataParallel(back_FCN)
back_FCN.to(device)
Discriminator = Discriminator()
Discriminator = nn.DataParallel(Discriminator)
Discriminator.to(device)
lr = 0.001

FCN_optim = optim.SGD(back_FCN.parameters(), lr = lr)
Discriminator_optim = optim.SGD(Discriminator.parameters(), lr = lr)

criterion = nn.BCELoss()
real_label = 1
fake_label = 0
epoch = 20

img_list = []
G_losses = []
D_losses = []
iters = 0

W1 = torch.rand(64, 64, requires_grad = True).to(device)
W2 = torch.rand(64, 64, requires_grad = True).to(device)

for epo in range(epoch):
    fig = plt.figure(figsize=(8, 4))
    plt.axis("off")
    for back, P_noback, P_withback in zip(Back_data, Portrait_noback_data, Portrait_withback_data):
        portrait_withback = P_withback[0].to(device)
        b_size = portrait_withback.size(0)
        label = torch.full((b_size,), real_label, device=device)
        output = Discriminator(portrait_withback).view(-1)
        real_error = criterion(output, label)
        Discriminator_optim.zero_grad()
        real_error.backward(retain_graph = True)
        D_x = output.mean().item()

        composit_portrait = W1 * back_FCN(back[0].to(device)) + W2 * P_noback[0].to(device)
        label.fill_(fake_label)
        output = Discriminator(composit_portrait.detach()).view(-1)
        fake_error = criterion(output, label)
        W1.retain_grad()
        W2.retain_grad()

        fake_error.backward(retain_graph=True)
        D_G_z1 = output.mean().item()
        err_D = real_error + fake_error
        Discriminator_optim.step()

        label.fill_(real_label)
        output = Discriminator(composit_portrait).view(-1)
        err_G = criterion(output, label)
        FCN_optim.zero_grad()
        err_G.backward(retain_graph = True)
        D_G_z2 = output.mean().item()
        FCN_optim.step()
        W1.data = W1.data - lr * W1.grad.data
        W2.data = W2.data - lr * W2.grad.data


        if (iters % 500 == 0) or ((epo == epoch - 1)):
            with torch.no_grad():
                fake = back_FCN(back[0].to(device)) + P_noback[0].to(device)
                fake = fake.detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))


        f = open('./portrait_background_composit.txt', 'a')
        f.write('\n[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epo, epoch, err_D.item(), err_G.item(), D_x, D_G_z1, D_G_z2))
        # print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
        #       % (epo, epoch, err_D.item(), err_G.item(), D_x, D_G_z1, D_G_z2))
        f.close()
        # Save Losses for plotting later
        G_losses.append(err_G.item())
        D_losses.append(err_D.item())

        iters += 1
    ims = [plt.imshow(np.transpose(img_list[-1], (1, 2, 0)), animated=True)]
    plt.savefig('./fake_images/{}.jpg'.format(epo))