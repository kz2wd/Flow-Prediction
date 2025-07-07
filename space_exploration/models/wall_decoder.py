import torch
import torch.nn.functional as F
from torch import nn

from space_exploration.models.StandardLearnModel import StandardLearnModel
from space_exploration.models.utils import ResBlockGen, UpSamplingBlock
from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace


class Generator(nn.Module):
    def __init__(self, y_dim, input_channels=3, n_residual_blocks=32, output_channels=3):
        super().__init__()
        self.y_dim = y_dim

        self.initial = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.up_sample_1 = UpSamplingBlock(64, 64)
        res_block = []
        up_sampling = []

        for index in range(n_residual_blocks):
            res_block.append(ResBlockGen(64))
            if index in [6, 12, 18, 24, 30]:
                res_block.append(UpSamplingBlock(64, 64))
                up_sampling.append(nn.Upsample(scale_factor=(1, 2, 1), mode='nearest'))

        self.res_block = nn.Sequential(*res_block)
        self.up_sampling = nn.Sequential(*up_sampling)

        self.conv2 = nn.Conv3d(64, 256, kernel_size=3, stride=1, padding=1)

        self.output_conv = nn.Conv3d(256, output_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.initial(x)
        if self.y_dim == 64:
            x = self.up_sample_1(x)
        up_samp = self.up_sampling(x)
        x = self.res_block(x)
        x = x + up_samp
        x = self.conv2(x)
        x = self.output_conv(x)
        return x

def get_decoder(y_dim):
    return Generator(y_dim, input_channels=3, n_residual_blocks=32, output_channels=3)

def loss_function(prediction, target):

    content_loss = F.mse_loss(prediction, target, reduction='mean')

    total_loss = content_loss # Add physic informed here
    return total_loss

class WallDecoder(StandardLearnModel):

    def init_components(self):
        self.decoder = get_decoder(self.prediction_sub_space.y[1])
        self.decoder.to(self.device)
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-4)
        self.loss = loss_function

    def __init__(self, name, prediction_sub_space: PredictionSubSpace):
        super().__init__(name, prediction_sub_space)
        self.prediction_sub_space = prediction_sub_space
        self.init_components()


