from abc import ABC, abstractmethod


class TransformBase(ABC):

    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target

    @abstractmethod
    def to_training(self, ds):
        pass

    @abstractmethod
    def from_training(self, ds):
        pass