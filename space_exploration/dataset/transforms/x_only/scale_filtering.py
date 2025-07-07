from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset.transforms.general.component_normalize import ComponentNormalize
from space_exploration.dataset.transforms.transformer_base import TransformBase
from scipy.ndimage import gaussian_filter
import numpy as np

def apply_DoG_batch_on_component(data, component, sigmas):
    blurs = [np.array([gaussian_filter(x, sigma=sigma) for x in data[:, component, ...]]) for sigma in sigmas]
    dog = [b2 - b1 for b1, b2 in zip(blurs, blurs[1:])]
    return dog


class ScaleFiltering(TransformBase):
    def __init__(self, dataset: Dataset, target):
        super().__init__(dataset, target)
        self.normalizer = ComponentNormalize(self.dataset, "X")

    def from_training(self, ds):
        return ds

    def to_training(self, ds):
        normalized_ds = self.normalizer.to_training(ds)
        # apply_DoG_batch_on_component()
        # TODO : Find which DoG to use!
