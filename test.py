import torch
import torch.nn as nn


img = torch.randn((1, 1, 5, 5))

print(img)

conv = nn.Conv2d(2, 2, 4)
conv_trans = nn.ConvTranspose2d(1, 1, padding=3, stride=1, dilation=1, kernel_size=10, output_padding=0)

print(conv_trans(img).shape)
