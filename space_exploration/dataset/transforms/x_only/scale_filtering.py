from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset.transforms.general.component_normalize import ComponentNormalize
from space_exploration.dataset.transforms.transformer_base import TransformBase
from scipy.ndimage import gaussian_filter
import numpy as np
import dask.array as da

def apply_DoG_batch_on_component(data, component, s1, s2):
    g1 = np.array([gaussian_filter(x, sigma=s1) for x in data[:, component, ...]])
    g2 = np.array([gaussian_filter(x, sigma=s2) for x in data[:, component, ...]])
    dog = g1 - g2
    return dog


class ScaleFiltering(TransformBase):
    def __init__(self, dataset: Dataset, target):
        super().__init__(dataset, target)
        self.normalizer = ComponentNormalize(self.dataset, "X")

    def from_training(self, ds):
        return ds

    def to_training(self, ds):
        normalized_ds = self.normalizer.to_training(ds).compute()

        # Sigmas obtained after performing Bayesian optimisation with Pearson correlation as evaluation for every pair
        # of input to output component (pressure + shear stress X/Z to U V W: 9 pairs) for every layer except first (63)
        # (63 x 9 optimisations performed with grid like strategy over 100 samples to find s1 and s2)
        # Performed clustering to regroup obtained sigmas to 8 values, could be improved more
        sigmas = [
            [0, (9.172, 26.348)],
            [0, (5.496, 29.237)],
            [0, (10.792, 16.214)],
            [0, (3.810, 23.111)],
            [0, (5.496, 29.237)],
            [0, (17.230, 26.774)],
            [0, (27.775, 29.892)],
            [0, (28.977, 35.500)],
            [1, (5.496, 29.237)],
            [1, (9.172, 26.348)],
            [1, (28.977, 35.500)],
            [2, (3.810, 23.111)],
            [2, (5.496, 29.237)],
            [2, (9.172, 26.348)],
            [2, (10.792, 16.214)],
        ]

        extra_comp = []

        for (comp, (s1, s2)) in sigmas:
            extra_comp.append(apply_DoG_batch_on_component(normalized_ds, comp, s1, s2))

        extra_comp = np.array(extra_comp).transpose((1, 0, 2, 3, 4))
        enriched_array = np.concatenate([normalized_ds, extra_comp], axis=1)
        return da.from_array(enriched_array)