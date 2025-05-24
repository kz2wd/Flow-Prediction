from abc import ABC, abstractmethod


class NormalizerBase(ABC):

    def __init__(self, dataset=None):
        pass

    @abstractmethod
    def normalize(self, ds):
        pass

    @abstractmethod
    def denormalize(self, ds):
        pass