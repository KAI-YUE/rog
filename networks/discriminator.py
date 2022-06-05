import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, UseInstanceNorm=False, UseSpectralNorm=True, ResBlocks=0):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels
        self.UseInstanceNorm = UseInstanceNorm
        self.UseSpectralNorm = UseSpectralNorm

        if self.UseInstanceNorm:
            self.part1 = nn.Sequential(
                self.spectral_norm(nn.Conv2d(self.in_channels, 64, 3, stride=2, padding=1)),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),

                self.spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1)),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                self.spectral_norm(nn.Conv2d(128, 256, 3, stride=1, padding=1)),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.part1 = nn.Sequential(
                self.spectral_norm(nn.Conv2d(self.in_channels, 64, 3, stride=2, padding=1)),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),

                self.spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                self.spectral_norm(nn.Conv2d(128, 256, 3, stride=1, padding=1)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),

                # self.spectral_norm(nn.Conv2d(256, 512, 3, stride=1, padding=1)),
                # nn.BatchNorm2d(512),
                # nn.LeakyReLU(0.2, inplace=True),
            )

        blocks = []
        for i in range(ResBlocks):
            blocks.append(ResnetBlock(512, 1, 'leaky_relu'))

        blocks.append(self.spectral_norm(nn.Conv2d(256, 1, 3, stride=1, padding=1)))
        blocks.append(nn.LeakyReLU(0.2, inplace=True))

        self.part2 = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        return torch.sigmoid(x)

    def spectral_norm(self, x):
        if (self.UseSpectralNorm):
            return spectral_norm(x)
        else:
            return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, activate='relu'):
        super(ResnetBlock, self).__init__()

        self.activate = activate
        self.basic_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation),
            nn.InstanceNorm2d(dim),
            self.relu(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.basic_block(x)
    
    def relu(self):
        if (self.activate == 'relu'):
            return nn.ReLU(inplace=True)
        elif (self.activate == 'leaky_relu'):
            return nn.LeakyReLU(0.2, inplace=True)