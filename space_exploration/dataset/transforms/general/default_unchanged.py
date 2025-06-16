from space_exploration.dataset.transforms.transformer_base import TransformBase


class DefaultUnchanged(TransformBase):
    def from_training(self, ds):
        return ds

    def to_training(self, ds):
        return ds