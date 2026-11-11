
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        def _block(in_channels, features,):
            return nn.Sequential(
                nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False,),
                nn.BatchNorm2d(num_features=features),
                nn.ReLU(),
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False,),
                nn.BatchNorm2d(num_features=features),
                nn.ReLU(),)

        features = init_features
        self.enc1 = _block(in_channels, features,)
        self.dnsamp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = _block(features, features * 2,)
        self.dnsamp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = _block(features * 2, features * 4,)
        self.dnsamp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = _block(features * 4, features * 8,)
        self.dnsamp4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _block(features * 8, features * 16,)

        self.upsamp4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.dec4 = _block((features * 8) * 2, features * 8,)
        self.upsamp3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3 = _block((features * 4) * 2, features * 4,)
        self.upsamp2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = _block((features * 2) * 2, features * 2,)
        self.upsamp1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = _block(features * 2, features,)

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.dnsamp1(enc1))
        enc3 = self.enc3(self.dnsamp2(enc2))
        enc4 = self.enc4(self.dnsamp3(enc3))

        bottleneck = self.bottleneck(self.dnsamp4(enc4))

        dec4 = self.upsamp4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec3 = self.upsamp3(self.dec4(dec4))
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec2 = self.upsamp2(self.dec3(dec3))
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec1 = self.upsamp1(self.dec2(dec2))
        dec1 = torch.cat((dec1, enc1), dim=1)
        return torch.sigmoid(self.conv(self.dec1(dec1)))


