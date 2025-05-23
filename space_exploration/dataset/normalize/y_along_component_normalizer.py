from space_exploration.dataset.ds_helper import y_along_component_denormalize, y_along_component_normalize
from space_exploration.dataset.normalize.normalizer_base import NormalizerBase


# Normalization used in the paper
class YAlongComponentNormalizer(NormalizerBase):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.w_stds = None
        self.v_stds = None
        self.u_stds = None
        self.w_means = None
        self.v_means = None
        self.u_means = None

    def denormalize(self, ds):
        return y_along_component_denormalize(ds, self.u_means, self.v_means, self.w_means,
                                             self.u_stds, self.v_stds, self.w_stds)

    def normalize(self, ds):
        return y_along_component_normalize(ds, self.u_means, self.v_means, self.w_means,
                                           self.u_stds, self.v_stds, self.w_stds)
