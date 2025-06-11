from abc import ABC, abstractmethod
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F

from space_exploration.FolderManager import FolderManager
from space_exploration.simulation_channel import SimulationChannel
from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace


def generator_loss(fake_y, y_pred, y_true):
    adversarial_labels = torch.ones_like(fake_y) - torch.rand_like(fake_y) * 0.2
    adversarial_loss = F.binary_cross_entropy(fake_y, adversarial_labels, reduction='mean')

    content_loss = F.mse_loss(y_pred, y_true, reduction='mean')

    total_loss = content_loss + 1e-3 * adversarial_loss
    # print(f"fake y mean: {fake_y.mean().item()}, y pred mean: {y_pred.mean().item()}, y true mean: {y_true.mean().item()}")
    # print(f"Content loss: {content_loss}, adversarial loss: {adversarial_loss}, total loss {total_loss}")
    return total_loss



def discriminator_loss(real_y, fake_y):
    real_labels = torch.ones_like(real_y) - torch.rand_like(real_y) * 0.2
    fake_labels = torch.rand_like(fake_y) * 0.2

    real_loss = F.binary_cross_entropy(real_y, real_labels, reduction='mean')
    fake_loss = F.binary_cross_entropy(fake_y, fake_labels, reduction='mean')

    total_loss = 0.5 * (real_loss + fake_loss)
    return total_loss

class GAN3D(ABC):
    def __init__(self, name, prediction_sub_space: PredictionSubSpace,
                 n_residual_blocks=32, input_channels=3, output_channels=3):



        self.prediction_sub_space = prediction_sub_space
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.name = name
        self.n_residual_blocks = n_residual_blocks

        FolderManager.init(self)  # ensure all related folders are created

        self.device = torch.device("cuda")  # if we cannot get cuda, don't even try...

        self.generator = self.get_generator(self.prediction_sub_space).to(self.device)
        self.discriminator = self.get_discriminator(self.prediction_sub_space).to(self.device)
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss


    @abstractmethod
    def get_generator(self, channel: SimulationChannel):
        pass

    @abstractmethod
    def get_discriminator(self, channel: SimulationChannel):
        pass

    def load(self, run_id):
        mlflow.set_tracking_uri("http://localhost:5000")
        checkpoint_path = "final_model/checkpoint_latest.pt"

        print(f"⌛ Fecthing artifact at {str(checkpoint_path)}")

        local_model_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=str(checkpoint_path))

        state_dict = torch.load(local_model_path, map_location="cuda")["generator_state_dict"]
        self.generator.load_state_dict(state_dict)
        self.generator.eval()
        self.generator.to(self.device)

        print(f"✅ Loaded generator from run {run_id}")

