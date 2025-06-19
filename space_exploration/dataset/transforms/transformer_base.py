from abc import ABC, abstractmethod


class TransformBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def to_training(self, ds):
        pass

    @abstractmethod
    def from_training(self, ds):
        pass