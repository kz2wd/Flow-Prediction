from space_exploration.dataset.ds_helper import y_along_component_denormalize, y_along_component_normalize
from space_exploration.dataset.transforms.transformer_base import TransformBase


# Normalization used in the paper
class YAlongComponentNormalizer(TransformBase):

    def __init__(self):
        super().__init__()


    def from_training(self, ds):
        return y_along_component_denormalize(ds)

    def to_training(self, ds):
        return y_along_component_normalize(ds)
