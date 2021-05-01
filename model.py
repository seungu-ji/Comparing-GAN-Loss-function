import torch
import torch.nn as nn

class DCGANgenrator(nn.Module):
    def __init__(self, in_channels, out_channels, nker):
        super(DCGANgenrator, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 8 * nker, 4, 1, 0, bias=False),
            nn.BatchNorm2d(8 * nker),
            nn.ReLU(True),

            nn.ConvTranspose2d(8 * nker, 4 * nker, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * nker),
            nn.ReLU(True),

            nn.ConvTranspose2d(4 * nker, 2 * nker, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * nker),
            nn.ReLU(True),

            nn.ConvTranspose2d(2 * nker, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)

class DCGANdiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker):
        super(DCGANdiscriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, nker, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nker, 2 * nker, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * nker),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(2 * nker, 4 * nker, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * nker),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(4 * nker, 8 * nker, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8 * nker),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8 * nker, out_channels, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)