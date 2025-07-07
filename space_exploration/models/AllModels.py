from enum import Enum

from space_exploration.models.UNet import SimpleUNet
from space_exploration.models.implementations.A import A as ModelA
from space_exploration.models.implementations.C import C as ModelC
from space_exploration.models.wall_decoder import WallDecoder
from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace


class ModelEnumBase(str, Enum):
    def __new__(cls, value, model):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.model = model
        return obj

    def __str__(self):
        return self.value


class ModelReferences(ModelEnumBase):
    A = ("A", ModelA)
    C = ("C", ModelC)
    WALL_DECODER = ("WALL_DECODER", lambda: WallDecoder("wall_decoder", PredictionSubSpace(y_end=64)))
    SIMPLE_UNET = ("SIMPLE_UNET", SimpleUNet)
