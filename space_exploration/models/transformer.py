import operator
from functools import reduce

import torch.nn as nn

from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace


class Transformer3D(nn.Module):
    def __init__(self, prediction_sub_space: PredictionSubSpace, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.sequence_length = reduce(operator.mul, prediction_sub_space.sizes(), 1)

        self.input_proj = nn.Linear(3, d_model)
        
