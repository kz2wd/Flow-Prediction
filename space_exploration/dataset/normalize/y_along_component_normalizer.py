from space_exploration.dataset.ds_helper import y_along_component_denormalize, y_along_component_normalize
from space_exploration.dataset.normalize.normalizer_base import NormalizerBase


# Normalization used in the paper
class YAlongComponentNormalizer(NormalizerBase):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.stats = dataset.get_stats()


    def denormalize(self, ds):
        return y_along_component_denormalize(ds, self.stats)

    def normalize(self, ds):
        return y_along_component_normalize(ds, self.stats)
