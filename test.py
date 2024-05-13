class Generator(nn.Module):
    def __init__(self, noise_size):
        super().__init__()

        self.lin1 = nn.Linear(noise_size, 1024) #[-1, 1, 32, 32]

        self.convtrans1 = nn.ConvTranspose2d(1, 8, stride=3, padding=0, output_padding=0,
                                              dilation=1, kernel_size=7, bias=False) #[-1, 8, 100, 100]
        self.convtrans2 = nn.ConvTranspose2d(8, 16, stride=2, padding=0, output_padding=0,
                                              dilation=1, kernel_size=2, bias=False) #[-1, 16, 200, 200]
        self.convtrans3 = nn.ConvTranspose2d(16, 16, stride=1, padding=0, output_padding=0,
                                            dilation=5, kernel_size=13, bias=False) #[-1, 16, 260, 260]
        self.conv1 =  nn.Conv2d(16, 8, stride=1, kernel_size=3, bias=False) #[-1, 1, 258, 258]
        self.conv2 =  nn.Conv2d(8, 3, stride=1, kernel_size=3, bias=False) #[-1, 1, 258, 258]

    def forward(self, x):
        x = self.lin1(x)

        x = x.view(-1, 1, 32, 32)

        x =self.convtrans1(x)
        x =self.convtrans2(x)
        x =self.convtrans3(x)
        x =self.conv1(x)
        x = self.conv2(x)

        image = x.view(-1, 3, 256, 256)
        
        return image