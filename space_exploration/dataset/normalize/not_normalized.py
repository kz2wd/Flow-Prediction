from space_exploration.dataset.normalize.normalizer_base import NormalizerBase


class NotNormalized(NormalizerBase):
    def denormalize(self, ds):
        return ds
    def normalize(self, ds):
        return ds
    