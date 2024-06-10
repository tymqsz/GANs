import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, noise_size):
        super().__init__()

        self.noise_size = noise_size

        self.convt1 = nn.ConvTranspose2d(noise_size, 64*8, 4, 1, 0, bias=False) #4
        self.bn1 = nn.BatchNorm2d(64*8)
        self.convt2 = nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False) #10
        self.bn2 = nn.BatchNorm2d(64*4)
        self.convt3 = nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False) #22
        self.bn3 = nn.BatchNorm2d(64*2)
        self.convt4 = nn.ConvTranspose2d(64*2, 64*2, 4, 2, 1, bias=False) #46
        self.bn4 = nn.BatchNorm2d(64*2)
        self.convt5 = nn.ConvTranspose2d(64*2, 3, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)

        """
        self.convt6 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        
        self.convt7 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        """
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()


    def forward(self, x):
        x = x.view(-1, self.noise_size, 1, 1)

        x = self.convt1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.convt2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.convt3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.convt4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.convt5(x)
        x = self.tanh(x)
        
        image = x.view(-1, 3, 64, 64)
        
        return image

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(512, 256, 4, 2, 1, bias=False)
        
        self.conv6 = nn.Conv2d(256, 1, 4, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)

        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        x = x.float()
        x = x.view(-1, 3, 64, 64)

        x = self.conv1(x)
        x = self.leakyrelu(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.leakyrelu(x)

        x = self.conv5(x)
        x = self.bn2(x)
        x = self.conv6(x)
        
        x = self.sigmoid(x)

        return x.view(-1, 1)
      

class GAN():
    def __init__(self, noise_size, image_dims, lr=2e-4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.noise_size = noise_size
        self.image_dims = image_dims

        self.generator = Generator(noise_size).to(self.device)
        self.discrimiantor = Discriminator().to(self.device)

        self.g_optim = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optim = optim.Adam(self.discrimiantor.parameters(), lr=lr, betas=(0.5, 0.999))

        self.loss_fn = nn.BCELoss()

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)
    
    def train(self, batch):   
        real_imgs, _ = batch 
        # Discriminator training
        self.discrimiantor.train()
        noise = torch.randn(real_imgs.shape[0], self.noise_size)
        noise = noise.to(device=self.device)

        self.generator.eval()
        with torch.no_grad():
            fake_imgs = self.generator(noise)
            fake_imgs.detach()

        fake_labels = torch.ones(real_imgs.shape[0], 1).to(device=self.device)
        real_labels = torch.zeros(real_imgs.shape[0], 1).to(device=self.device)

        yp = self.discrimiantor(fake_imgs)
        fake_loss = self.loss(y_pred=yp, y_true=fake_labels)
        real_loss = self.loss(y_pred=self.discrimiantor(real_imgs.to(device=self.device)), y_true=real_labels)

        loss = (fake_loss + real_loss)/2
        self.d_optim.zero_grad()
        loss.backward()
        self.d_optim.step()

        # Generator training
        self.generator.train()
        noise = torch.randn(real_imgs.shape[0], self.noise_size)
        noise = noise.to(device=self.device)

        desired_labels = torch.zeros(real_imgs.shape[0], 1)
        desired_labels = desired_labels.to(device=self.device)

        labels = self.discrimiantor(self.generator(noise))

        loss = self.loss(labels, desired_labels)
        self.g_optim.zero_grad()
        loss.backward()
        self.g_optim.step()
       # first zero grad + no self.discirmintor.eval()
    
    def plot_samples(self, epoch, n_samples=3, noise=None):
        if noise is None:
            noise = torch.randn(n_samples, self.noise_size)
            noise = noise.to(device=self.device)
        else:
            noise = noise.to(device=self.device)
            
        
        with torch.no_grad():
            images = self.generator(noise)

        fig = plt.figure(figsize=(10, 6))

        for i, image in enumerate(images):
            ax = fig.add_subplot(1, 3, i+1)

            img = np.transpose(image.cpu().detach(), (1,2, 0))
            ax.imshow(img, cmap="gray", norm="linear")
            if i == 0:
                ax.set_title(f"epoch {epoch}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()      