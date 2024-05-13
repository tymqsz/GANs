import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, noise_size):
        super().__init__()

        self.lin1 = nn.Linear(noise_size, 1024) #[-1, 1, 32, 32]

        self.convtrans1 = nn.ConvTranspose2d(1, 16, stride=3, padding=0, output_padding=0,
                                              dilation=1, kernel_size=7, bias=False) #[-1, 8, 100, 100]
        self.convtrans2 = nn.ConvTranspose2d(16, 32, stride=2, padding=0, output_padding=0,
                                              dilation=1, kernel_size=2, bias=False) #[-1, 16, 200, 200]
        self.convtrans3 = nn.ConvTranspose2d(32, 32, stride=1, padding=0, output_padding=0,
                                            dilation=5, kernel_size=13, bias=False) #[-1, 16, 260, 260]
        self.conv1 =  nn.Conv2d(32, 32, stride=1, kernel_size=3, bias=False) #[-1, 1, 258, 258]
        self.conv2 =  nn.Conv2d(32, 1, stride=1, kernel_size=3, bias=False) #[-1, 1, 258, 258]

    def forward(self, x):
        x = F.relu(self.lin1(x))

        x = x.view(-1, 1, 32, 32)

        x = F.leaky_relu(self.convtrans1(x), 0.02)
        x = F.leaky_relu(self.convtrans2(x), 0.02)
        x = F.leaky_relu(self.convtrans3(x), 0.02)
        x = F.leaky_relu(self.conv1(x), 0.02)
        x = self.conv2(x)
        image = x.view(-1, 1, 256, 256)
        
        return image
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        def conv_stack(in_channels, out_channels, ks_conv, ks_pool):
            stack = [nn.Conv2d(in_channels, out_channels, kernel_size=ks_conv, stride=1, bias=False)]

            stack.append(nn.MaxPool2d(kernel_size=ks_pool, stride=1))
            stack.append(nn.LeakyReLU(0.2))
            stack.append(nn.Dropout2d())
            
            return stack
        
        self.cnn = nn.Sequential(
            *conv_stack(1, 32, 3, 3), #[-1, 8, 252, 254]
            *conv_stack(32, 16, 5, 5), #[-1, 16, 244, 244]
            *conv_stack(16, 1, 3, 3) #[-1, 1, 240, 240]
        )

        self.lin1 = nn.Linear(in_features=1*240*240, out_features=1024) 
        self.lin2 = nn.Linear(in_features=1024, out_features=256)
        self.lin3 = nn.Linear(in_features=256, out_features=1) 

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        x = x.view(-1, 1, 256, 256)
        
        x = self.cnn(x)

        x = x.view(-1, 1*240*240)

        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.lin3(x)

        return self.sigmoid(x)

class GAN():
    def __init__(self, noise_size, image_dims, lr=1e-4):
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

        fake_labels = torch.zeros(real_imgs.shape[0], 1).to(device=self.device)
        real_labels = torch.ones(real_imgs.shape[0], 1).to(device=self.device)

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

        desired_labels = torch.ones(real_imgs.shape[0], 1)
        desired_labels = desired_labels.to(device=self.device)

        labels = self.discrimiantor(self.generator(noise))

        loss = self.loss(labels, desired_labels)
        self.g_optim.zero_grad()
        loss.backward()
        self.g_optim.step()
       
       # first zero grad + no self.discirmintor.eval()
    
    def plot_samples(self, n_samples=3):
        noise = torch.randn(n_samples, self.noise_size)
        noise = noise.to(device=self.device)
        

        images = self.generator(noise)
        
        for image in images:
            plt.figure()
            plt.title("generated image")

            plt.imshow(image.detach().cpu().view(256, 256), cmap="gray")

        plt.show()        
        self.generator.train()