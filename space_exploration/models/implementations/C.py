from torch import nn

from space_exploration.models.PaperBase import PaperBase
from space_exploration.models.utils import ResBlockGen, UpSamplingBlock
from space_exploration.simulation_channel import SimulationChannel
from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace


class Generator(nn.Module):
    def __init__(self, prediction_sub_space: PredictionSubSpace, input_channels=3, n_residual_blocks=32, output_channels=3):
        super().__init__()
        self.ny = prediction_sub_space.y[1]

        self.initial = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # self.up_sample_1 = UpSamplingBlock(64, 64)
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
        # x = self.up_sample_1(x)
        up_samp = self.up_sampling(x)
        x = self.res_block(x)
        x = x + up_samp
        x = self.conv2(x)
        x = self.output_conv(x)
        return x

class C(PaperBase):
    def __init__(self, name):
        super().__init__(name, PredictionSubSpace(y_end=32))

    def get_generator(self, prediction_sub_space: PredictionSubSpace):
        return Generator(prediction_sub_space)