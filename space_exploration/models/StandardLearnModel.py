import time
from abc import abstractmethod

import dask.array as da
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from space_exploration.models.model_base import PredictionModel
from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace


class StandardLearnModel(PredictionModel):

    def __init__(self, name, prediction_sub_space: PredictionSubSpace):
        super().__init__(name, prediction_sub_space)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.decoder = None
        self.optimizer = None
        self.loss = None

    @abstractmethod
    def init_components(self):
        pass

    def _train_one_epoch(self, profiler=None):
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

            if profiler is not None:
                profiler.step()

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

    def train_cycle(self, epoch, start_time, profiler=None):

        train_loss = self._train_one_epoch(profiler)
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

    def get_all_torch_components_named(self):
        return {"model": self.decoder}

