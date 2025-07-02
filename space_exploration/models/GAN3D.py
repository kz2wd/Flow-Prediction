import time
from abc import abstractmethod

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import dask.array as da

from space_exploration.FolderManager import FolderManager
from space_exploration.models.model_base import PredictionModel
from space_exploration.simulation_channel import SimulationChannel
from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace


def generator_loss(fake_y, y_pred, y_true):
    adversarial_labels = torch.ones_like(fake_y) - torch.rand_like(fake_y) * 0.2
    adversarial_loss = F.binary_cross_entropy(fake_y, adversarial_labels, reduction='mean')

    content_loss = F.mse_loss(y_pred, y_true, reduction='mean')

    total_loss = content_loss + 1e-3 * adversarial_loss
    return total_loss


def discriminator_loss(real_y, fake_y):
    real_labels = torch.ones_like(real_y) - torch.rand_like(real_y) * 0.2
    fake_labels = torch.rand_like(fake_y) * 0.2

    real_loss = F.binary_cross_entropy(real_y, real_labels, reduction='mean')
    fake_loss = F.binary_cross_entropy(fake_y, fake_labels, reduction='mean')

    total_loss = 0.5 * (real_loss + fake_loss)
    return total_loss


class GAN3D(PredictionModel):
    def __init__(self, name, prediction_sub_space: PredictionSubSpace,
                 n_residual_blocks=32, input_channels=3, output_channels=3):
        super().__init__(name, prediction_sub_space)

        self.input_channels = input_channels
        self.output_channels = output_channels
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

    def load(self, state_dict):

        self.generator.load_state_dict(state_dict["generator_state_dict"])
        self.discriminator.load_state_dict(state_dict["discriminator_state_dict"])
        self.discriminator.eval()
        self.discriminator.to(self.device)

        self.generator.eval()
        self.generator.to(self.device)

    def custom_load(self, run_id, artifact_path):
        mlflow.set_tracking_uri("http://localhost:5000")
        print(f"⌛ Fetching remote artifact at {str(artifact_path)}")
        local_model_path = mlflow.artifacts.download_artifacts(run_id=run_id,
                                                               artifact_path=str(artifact_path))
        state_dict = torch.load(local_model_path, map_location="cuda")
        self.load(state_dict)

    def _train_one_epoch(self):
        self.generator.train()
        self.discriminator.train()
        gen_losses, disc_losses = [], []

        for x_target, y_target in tqdm.tqdm(self.train_ds, desc="Training"):
            x_target, y_target = x_target.to(self.device), y_target.to(self.device)

            # === Generator forward and loss ===
            y_pred = self.generator(x_target)
            with torch.no_grad():
                fake_output_for_gen = self.discriminator(y_pred)
            gen_loss = self.generator_loss(fake_output_for_gen, y_pred, y_target)

            self.generator_optimizer.zero_grad()
            gen_loss.backward()
            self.generator_optimizer.step()

            # === Discriminator forward and loss ===
            real_output = self.discriminator(y_target)
            fake_output = self.discriminator(y_pred.detach())
            disc_loss = self.discriminator_loss(real_output, fake_output)

            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            self.discriminator_optimizer.step()

            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())

            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                print("NaN or Inf detected in generator output!")

        return np.mean(gen_losses), np.mean(disc_losses)

    def _validate(self):
        self.generator.eval()
        self.discriminator.eval()
        gen_losses, disc_losses = [], []

        with torch.no_grad():
            for x_target, y_target in tqdm.tqdm(self.val_ds, desc="Validating"):
                x_target, y_target = x_target.to(self.device), y_target.to(self.device)
                y_pred = self.generator(x_target)
                real_output = self.discriminator(y_target)
                fake_output = self.discriminator(y_pred)

                gen_loss = self.generator_loss(fake_output, y_pred, y_target)
                disc_loss = self.discriminator_loss(real_output, fake_output)

                gen_losses.append(gen_loss.item())
                disc_losses.append(disc_loss.item())

        return np.mean(gen_losses), np.mean(disc_losses)

    def train_cycle(self, epoch, start_time):

        train_gen_loss, train_disc_loss = self._train_one_epoch()
        val_gen_loss, val_disc_loss = self._validate()

        elapsed = time.time() - start_time

        # Log metrics
        mlflow.log_metrics({
            "train_gen_loss": train_gen_loss,
            "train_disc_loss": train_disc_loss,
            "val_gen_loss": val_gen_loss,
            "val_disc_loss": val_disc_loss
        }, step=epoch)

        print(f"[Epoch {epoch}] "
              f"train_gen_loss={train_gen_loss:.4f}, "
              f"train_disc_loss={train_disc_loss:.4f}, "
              f"val_gen_loss={val_gen_loss:.4f}, "
              f"val_disc_loss={val_disc_loss:.4f}, "
              f"time={elapsed:.2f}s")

        self.lr_scheduler.step(val_gen_loss)

        return val_gen_loss

    def prepare_train(self, train_ds, val_ds, test_ds):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.generator_optimizer, mode='min', factor=0.5, patience=2
        )

    def save(self, epoch, ckpt):
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.discriminator_optimizer.state_dict()
        }, ckpt)

    def training_end(self):
        self.generator.eval()

        mse_losses = []

        with torch.no_grad():
            for x_target, y_target in tqdm.tqdm(self.test_ds, desc="Testing"):
                x_target, y_target = x_target.to(self.device), y_target.to(self.device)
                y_pred = self.generator(x_target)
                mse_loss = F.mse_loss(y_pred, y_target).item()
                mse_losses.append(mse_loss)

        mean_mse = np.mean(mse_losses)
        print(f"[Test MSE] {mean_mse:.6f}")
        mlflow.log_metric("test_mse", float(mean_mse))

    def predict(self, dataset):
        self.generator.eval()

        predictions = []
        with torch.no_grad():
            for x_target, y_target in tqdm.tqdm(dataset, desc="Testing"):
                x_target, y_target = x_target.to(self.device), y_target.to(self.device)
                y_pred = self.generator(x_target)
                predictions.append(y_pred.cpu().numpy())

        ds = da.concatenate(predictions, axis=0)
        return ds.rechunk()
