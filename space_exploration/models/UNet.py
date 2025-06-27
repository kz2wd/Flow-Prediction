import torch
import torch.nn as nn


def double_convolution(in_channels, out_channels):
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv_op


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = double_convolution(3, 64)
        self.conv2 = double_convolution(64, 128)
        self.conv3 = double_convolution(128, 256)
        self.conv4 = double_convolution(256, 512)
        self.conv5 = double_convolution(512, 1024)

        self.up_transpose1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.up_conv1 = double_convolution(1024, 512)
        self.up_transpose2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up_conv2 = double_convolution(512, 256)
        self.up_transpose3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up_conv3 = double_convolution(256, 128)
        self.up_transpose4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.up_conv4 = double_convolution(128, 64)

        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        down1 = self.conv1(x)
        down = self.max_pool2d(down1)
        down2 = self.conv2(down)
        down = self.max_pool2d(down2)
        down3 = self.conv3(down)
        down = self.max_pool2d(down3)
        down4 = self.conv4(down)
        down = self.max_pool2d(down4)
        down5 = self.conv5(down)

        up1 = self.up_transpose1(down5)
        x = self.up_conv1(torch.cat((down4, up1), dim=1))
        up2 = self.up_transpose2(x)
        x = self.up_conv2(torch.cat((down3, up2), dim=1))
        up3 = self.up_transpose3(x)
        x = self.up_conv3(torch.cat((down2, up3), dim=1))
        up4 = self.up_transpose4(x)
        x = self.up_conv4(torch.cat((down1, up4), dim=1))
        out = self.out(x)
        return out


