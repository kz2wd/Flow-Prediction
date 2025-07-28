from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset.transforms.general.component_normalize import ComponentNormalize
from space_exploration.dataset.transforms.transformer_base import TransformBase
from scipy.ndimage import gaussian_filter
import numpy as np

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
        normalized_ds = self.normalizer.to_training(ds)

        # Some sigmas chosen from first search
        sigmas = [
            [0, (9.60, 24.35)],
            [1, (19.62, 28.77)],
            [1, (24.12, 25.92)],
        ]

        extra_comp = []

        for (comp, (s1, s2)) in sigmas:
            extra_comp.append(apply_DoG_batch_on_component(normalized_ds, comp, s1, s2))

        extra_comp = np.array(extra_comp).transpose((1, 0, 2, 3, 4))
        enriched_array = np.concatenate([normalized_ds, extra_comp], axis=1)
        return enriched_array