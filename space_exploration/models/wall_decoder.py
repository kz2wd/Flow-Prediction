import time

import mlflow
import numpy as np
import torch
import tqdm
import torch.nn.functional as F

from space_exploration.models.model_base import PredictionModel

import dask.array as da
from torch import nn

from space_exploration.models.utils import ResBlockGen, UpSamplingBlock
from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace


class Generator(nn.Module):
    def __init__(self, y_dim, input_channels=3, n_residual_blocks=32, output_channels=3):
        super().__init__()
        self.y_dim = y_dim

        self.initial = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.up_sample_1 = UpSamplingBlock(64, 64)
        res_block = []
        up_sampling = []

        for index in range(n_residual_blocks):
            res_block.append(ResBlockGen(64))
            if index in [6, 12, 18, 24, 30]:
                res_block.append(UpSamplingBlock(64, 64))
                up_sampling.append(nn.Upsample(scale_factor=(1, 2, 1), mode='nearest'))

        self.res_block = nn.Sequential(*res_block)
        self.up_sampling = nn.Sequential(*up_sampling)

        self.conv2 = nn.Conv3d(64, 256, kernel_size=3, stride=1, padding=1)

        self.output_conv = nn.Conv3d(256, output_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.initial(x)
        if self.y_dim == 64:
            x = self.up_sample_1(x)
        up_samp = self.up_sampling(x)
        x = self.res_block(x)
        x = x + up_samp
        x = self.conv2(x)
        x = self.output_conv(x)
        return x

def get_decoder(y_dim):
    return Generator(y_dim, input_channels=3, n_residual_blocks=32, output_channels=3)

def loss_function(prediction, target):

    content_loss = F.mse_loss(prediction, target, reduction='mean')

    total_loss = content_loss # Add physic informed here
    return total_loss

class WallDecoder(PredictionModel):

    def __init__(self, name, prediction_sub_space: PredictionSubSpace):
        super().__init__(name, prediction_sub_space)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.decoder = get_decoder(prediction_sub_space.y[1])
        self.decoder.to(self.device)
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-4)
        self.loss = loss_function

    def _train_one_epoch(self):
        self.decoder.train()
        losses = []

        for x_target, y_target in tqdm.tqdm(self.train_ds, desc="Training"):
            x_target, y_target = x_target.to(self.device), y_target.to(self.device)

            # === Generator forward and loss ===
            y_pred = self.decoder(x_target)
            loss = self.loss(y_pred, y_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                print("NaN or Inf detected in output!")

        return np.mean(losses)

    def _validate(self):
        self.decoder.eval()
        losses = []

        with torch.no_grad():
            for x_target, y_target in tqdm.tqdm(self.val_ds, desc="Validating"):
                x_target, y_target = x_target.to(self.device), y_target.to(self.device)
                y_pred = self.decoder(x_target)

                loss = self.loss(y_pred, y_target)

                losses.append(loss.item())

        return np.mean(losses)

    def train_cycle(self, epoch, start_time):

        train_loss = self._train_one_epoch()
        val_loss = self._validate()

        elapsed = time.time() - start_time

        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, step=epoch)

        print(f"[Epoch {epoch}] "
              f"train_gen_loss={train_loss:.4f}, "
              f"val_gen_loss={val_loss:.4f}, "
              f"time={elapsed:.2f}s")

        self.lr_scheduler.step(val_loss)

        return val_loss

    def save(self, epoch, ckpt):
        torch.save({
            'epoch': epoch,
            'decoder': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, ckpt)

    def load(self, state_dict):
        self.decoder.load_state_dict(state_dict["decoder"])
        self.decoder.eval()
        self.decoder.to(self.device)
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def prepare_train(self, train_ds, val_ds, test_ds):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )

    def training_end(self):
        self.decoder.eval()
        mse_losses = []
        with torch.no_grad():
            for x_target, y_target in tqdm.tqdm(self.test_ds, desc="Testing"):
                x_target, y_target = x_target.to(self.device), y_target.to(self.device)
                y_pred = self.decoder(x_target)
                mse_loss = F.mse_loss(y_pred, y_target).item()
                mse_losses.append(mse_loss)

        mean_mse = np.mean(mse_losses)
        mlflow.log_metric("test_mse", float(mean_mse))

    def predict(self, dataset):
        self.decoder.eval()

        predictions = []
        with torch.no_grad():
            for x_target, y_target in tqdm.tqdm(dataset, desc="Testing"):
                x_target, y_target = x_target.to(self.device), y_target.to(self.device)
                y_pred = self.decoder(x_target)
                predictions.append(y_pred.cpu().numpy())

        ds = da.concatenate(predictions, axis=0)
        return ds.rechunk()

