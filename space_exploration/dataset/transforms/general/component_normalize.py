import dask.array as da



from space_exploration.dataset.transforms.transformer_base import TransformBase


class ComponentNormalize(TransformBase):
    def from_training(self, ds):
        sample_size = 100

        means = ds[sample_size:].mean(axis=(0, 2, 3, 4)).compute()
        stds = ds[sample_size:].std(axis=(0, 2, 3, 4)).compute()
        stds = da.where(stds == 0, 1e-8, stds)  # avoid division by zero

        means = means[None, :, None, None, None]
        stds = stds[None, :, None, None, None]

        return (ds - means) / stds

    def to_training(self, ds):
        sample_size = 100

        means = ds[sample_size:].mean(axis=(0, 2, 3, 4)).compute()
        stds = ds[sample_size:].std(axis=(0, 2, 3, 4)).compute()

        # Reshape for broadcasting: (C,) → (1, C, 1, 1, 1)
        means = means[None, :, None, None, None]
        stds = stds[None, :, None, None, None]
        stds = da.where(stds == 0, 1e-8, stds)  # avoid division by zero

        return (ds - means) / stds
