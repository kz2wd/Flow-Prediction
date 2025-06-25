from dask.diagnostics import ProgressBar
import dask.array as da

from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset import s3_access
from space_exploration.dataset.ds_helper import y_along_component_denormalize, y_along_component_normalize
from space_exploration.dataset.transforms.transformer_base import TransformBase


# Normalization used in the paper
class YAlongComponentNormalizer(TransformBase):

    def __init__(self, dataset: Dataset, target):
        super().__init__(dataset, target)

        print(f"Loading stds & means of dataset {dataset.name}")
        self.name = "YAlongComponentNormalizer"
        mean_and_std = s3_access.get_ds(dataset.get_transform_data_path(self.name, self.target))

        if mean_and_std is None:

            if dataset.is_generated:
                raise Exception(f"Cannot compute mean and std from a generated dataset, please first load this "
                                f"transformer with a standard dataset. Reason: Missing file at "
                                f"[{dataset.get_transform_data_path(self.name, self.target)}]")

            print("No mean and std data found, computing them")
            ds = dataset.y
            with ProgressBar():
                means = ds.mean(axis=(0, 2, 4)).compute()
                stds = ds.std(axis=(0, 2, 4)).compute()
            mean_and_std = da.stack([means, stds])
            print(f"Saving freshly computed mean and std")
            s3_access.store_ds(mean_and_std, dataset.get_transform_data_path(self.name, self.target))

        self.mean_and_std = mean_and_std

    def from_training(self, ds):
        mean, std = self.mean_and_std
        return y_along_component_denormalize(ds, mean, std)

    def to_training(self, ds):
        mean, std = self.mean_and_std
        return y_along_component_normalize(ds, mean, std)
