
import dask.array as da

from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset.transforms.general.component_normalize import ComponentNormalize
from space_exploration.dataset.transforms.transformer_base import TransformBase


class UNetAdapter64(TransformBase):

    def __init__(self, dataset: Dataset, target):
        super().__init__(dataset, target)
        self.normalizer = ComponentNormalize(self.dataset, "X")

    def to_training(self, ds):
        B, C, X, Y, Z = ds.shape
        assert C == 3 and Y == 1, "Expected input shape (B, 3, X, 1, Z)"

        ds = self.normalizer.to_training(ds)

        # Repeat along Y-axis: from (B, 3, X, 1, Z) → (B, 3, X, 64, Z)
        expanded = da.tile(ds, (1, 1, 1, 64, 1))  # repeat Y=1 → Y=64

        # Create zero velocity components: (B, 3, X, 64, Z)
        zeros = da.zeros_like(expanded)

        # Concatenate: (B, 6, X, 64, Z)
        return da.concatenate([zeros, expanded], axis=1)


    def from_training(self, ds):
        raise Exception("Not implemented")