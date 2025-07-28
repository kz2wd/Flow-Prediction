from space_exploration.models.model_base import PredictionModel


class HGUnet(PredictionModel):
    def get_all_torch_components_named(self):
        pass

    def predict(self, dataset):
        pass

    def load(self, state_dict):
        pass

    def save(self, epoch, ckpt):
        pass

    def training_end(self):
        pass

    def prepare_train(self, train_ds, val_ds, test_ds):
        pass

    def train_cycle(self, epoch, start_time, profiler=None):
        pass