from space_exploration.dataset.ds_helper import y_along_component_denormalize, y_along_component_normalize
from space_exploration.dataset.transforms.transformer_base import TransformBase


# Normalization used in the paper
class YAlongComponentNormalizer(TransformBase):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.stats = dataset.get_stats()


    def from_training(self, ds):
        return y_along_component_denormalize(ds, self.stats)

    def to_training(self, ds):
        return y_along_component_normalize(ds, self.stats)
