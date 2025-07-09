from abc import ABC, abstractmethod

from space_exploration.beans.prediction_subspace_bean import PredictionSubSpace


class PredictionModel(ABC):
    def __init__(self, name, prediction_sub_space: PredictionSubSpace):
        self.prediction_sub_space = prediction_sub_space
        self.name = name

    @abstractmethod
    def train_cycle(self, epoch, start_time):
        pass

    @abstractmethod
    def prepare_train(self, train_ds, val_ds, test_ds):
        pass

    @abstractmethod
    def training_end(self):
        pass

    @abstractmethod
    def save(self, epoch, ckpt):
        pass

    @abstractmethod
    def load(self, state_dict):
        pass

    @abstractmethod
    def predict(self, dataset):
        pass

    @abstractmethod
    def get_all_torch_components_named(self):
        pass

    def print_stats(self):
        print("name\ttotal parameters\ttraining parameters")
        for name, component in self.get_all_torch_components_named().items():
            total_params = sum(p.numel() for p in component.parameters())
            total_trainable_params = sum(
                p.numel() for p in component.parameters() if p.requires_grad)
            print(f"{name}\t{total_params:,}\t{total_trainable_params:,}")




