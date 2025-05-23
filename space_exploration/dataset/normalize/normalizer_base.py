from abc import ABC, abstractmethod


class NormalizerBase(ABC):

    def __init__(self, dataset):
        pass

    @abstractmethod
    def normalize(self, ds):
        pass

    @abstractmethod
    def denormalize(self, ds):
        pass