from torch import nn


class ResBlockGen(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.prelu(out)
        out = self.conv2(out)
        out = out + identity
        return out

class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=(1, 2, 1), mode='nearest')
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.prelu(x)
        return x


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

