from abc import ABC, abstractmethod

from mlflow import keras

from space_exploration.simulation_channel import SimulationChannel


class GAN3D(ABC):
    def __init__(self, model_name, checkpoint, channel: SimulationChannel,
                 n_residual_blocks=32, input_channels=3, output_channels=3, learning_rate=1e-4, ):
        # Data centered, normalized & scaled
        self.target_x = None
        self.target_y = None
        self.predicted_y = None

        self.checkpoint = checkpoint

        self.channel = channel

        self.learning_rate = learning_rate
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.model_name = model_name
        self.n_residual_blocks = n_residual_blocks

    @abstractmethod
    def generator(self):
        pass

    @abstractmethod
    def discriminator(self):
        pass

    @abstractmethod
    def generator_loss(self):
        pass

    @abstractmethod
    def discriminator_loss(self):
        pass

    def generator_optimizer(self):
        return keras.optimizers.Adam(learning_rate=self.learning_rate)

    def discriminator_optimizer(self):
        return keras.optimizers.Adam(learning_rate=self.learning_rate)
