import shutil
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import tqdm

from space_exploration.FolderManager import FolderManager
from space_exploration.beans.dataset_bean import Dataset
from space_exploration.beans.training_bean import Training
from space_exploration.dataset import s3_access
from space_exploration.dataset.db_access import global_session
from space_exploration.dataset.transforms.AllTransforms import TransformationReferences
from space_exploration.models.AllModels import ModelReferences
from space_exploration.training_utils import get_split_datasets, prepare_dataset, get_prediction_ds, test_gan


class ModelTraining:
    def __init__(self, model_name, dataset_name, x_transform_name, y_transform_name, batch_size, data_amount=-1, max_epochs=50, saving_freq=3, train_patience=4, name=None, bean=None):

        self.train_patience = train_patience
        self.saving_freq = saving_freq
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.y_transform_name = y_transform_name
        self.x_transform_name = x_transform_name
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.bean = bean

        self.model_ref = ModelReferences(model_name)

        self.model = self.model_ref.model()
        self.device = self.model.device

        self.dataset = Dataset.get_dataset_or_fail(dataset_name)

        self.x_transform_ref = TransformationReferences(x_transform_name)
        self.y_transform_ref = TransformationReferences(y_transform_name)

        self.data_amount = data_amount
        if self.data_amount == -1:
            self.data_amount = self.dataset.size

        self.name = name
        if self.name is None:
            self.name = f"{model_name}_{dataset_name}_{data_amount}_{batch_size}"

        # === ARTIFACT MANAGEMENT ===
        self.artifact_dir = Path(FolderManager.artifact_backup_folder(self.model))
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.latest_ckpt = self.artifact_dir / "checkpoint_latest.pt"
        self.best_ckpt = self.artifact_dir / "checkpoint_best.pt"

    def run(self):
        mlflow.set_tracking_uri("http://localhost:5000")
        with mlflow.start_run(run_name=self.name):

            run = mlflow.active_run()
            self._record_in_database(run.info.run_id)

            self._prepare_train()

            try:
                self._internal_train()
            except KeyboardInterrupt as e:
                print(f"🚩 Stopping training due to user request")
            except Exception as e:
                print(f"Encountered exception while training: {e}")

            finally:
                self._upload_best_model()
                self._test_model()
                self._clean_training()

    def _prepare_train(self):
        y_dim = self.model.prediction_sub_space.y[1]
        ds = self.dataset.get_training_dataset(y_dim, self.x_transform_ref.transformation, self.y_transform_ref.transformation, self.data_amount)
        self.train_ds, self.val_ds, self.test_ds = get_split_datasets(ds, batch_size=4, val_ratio=0.2, test_ratio=0.05,
                                                 device=self.model.device)



    def _clean_training(self):
        print(f"🔄️ Deleting local backups in [{self.artifact_dir}]")
        shutil.rmtree(self.artifact_dir)

    def _train_one_epoch(self):
        self.model.generator.train()
        self.model.discriminator.train()
        gen_losses, disc_losses = [], []

        for x_target, y_target in tqdm.tqdm(self.train_ds, desc="Training"):
            x_target, y_target = x_target.to(self.device), y_target.to(self.device)

            # === Generator forward and loss ===
            y_pred = self.model.generator(x_target)
            with torch.no_grad():
                fake_output_for_gen = self.model.discriminator(y_pred)
            gen_loss = self.model.generator_loss(fake_output_for_gen, y_pred, y_target)

            self.model.generator_optimizer.zero_grad()
            gen_loss.backward()
            self.model.generator_optimizer.step()

            # === Discriminator forward and loss ===
            real_output = self.model.discriminator(y_target)
            fake_output = self.model.discriminator(y_pred.detach())
            disc_loss = self.model.discriminator_loss(real_output, fake_output)

            self.model.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            self.model.discriminator_optimizer.step()

            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())

            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                print("NaN or Inf detected in generator output!")

        return np.mean(gen_losses), np.mean(disc_losses)

    def _validate(self):
        self.model.generator.eval()
        self.model.discriminator.eval()
        gen_losses, disc_losses = [], []

        with torch.no_grad():
            for x_target, y_target in tqdm.tqdm(self.val_ds, desc="Validating"):
                x_target, y_target = x_target.to(self.device), y_target.to(self.device)
                y_pred = self.model.generator(x_target)
                real_output = self.model.discriminator(y_target)
                fake_output = self.model.discriminator(y_pred)

                gen_loss = self.model.generator_loss(fake_output, y_pred, y_target)
                disc_loss = self.model.discriminator_loss(real_output, fake_output)

                gen_losses.append(gen_loss.item())
                disc_losses.append(disc_loss.item())

        return np.mean(gen_losses), np.mean(disc_losses)

    def _save_model(self, epoch, ckpt):
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.model.generator_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.model.discriminator_optimizer.state_dict()
        }, ckpt)

    def _upload_best_model(self):
        print("🕓 Uploading best model")
        mlflow.log_artifact(str(self.best_ckpt), artifact_path="final_model")

    def _internal_train(self):
        # === OPTIMIZERS AND LR SCHEDULERS ===
        self.model.generator_optimizer = torch.optim.Adam(self.model.generator.parameters(), lr=1e-4)
        self.model.discriminator_optimizer = torch.optim.Adam(self.model.discriminator.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.model.generator_optimizer, mode='min', factor=0.5, patience=2
        )

        # === EARLY STOPPING SETUP ===
        best_val_loss = float('inf')
        patience_counter = 0


        mlflow.set_tag("model_type", "GAN")
        mlflow.log_params({
            "max_epochs": self.max_epochs,
            "model_name": self.model.name,
        })

        start_time = time.time()
        for epoch in range(1, self.max_epochs + 1):
            print(f"\nEpoch {epoch}/{self.max_epochs}")
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

            lr_scheduler.step(val_gen_loss)

            if epoch % self.saving_freq == 0:
                self._save_model(epoch, self.latest_ckpt)
                print(f"Saving checkpoint for epoch {epoch}...")

            if val_gen_loss < best_val_loss:
                best_val_loss = val_gen_loss
                patience_counter = 0
                self._save_model(epoch, self.best_ckpt)
                print(f"Saving best model at epoch {epoch}...")
            else:
                patience_counter += 1
                print(f"  [EarlyStopping] Patience {patience_counter}/{self.train_patience}")
                if patience_counter >= self.train_patience:
                    print("  Early stopping triggered.")
                    break

    def _record_in_database(self, run_id):
        training = Training(
            dataset=self.dataset,
            data_amount=self.data_amount,
            batch_size=self.batch_size,
            model=str(self.model_ref),
            x_transform=str(self.x_transform_ref),
            y_transform=str(self.y_transform_ref),
            run_id=run_id,
        )
        global_session.add(training)
        global_session.commit()

        self.bean = training


    def load_model(self):
        if self.bean is None:
            raise Exception("No existing training record!")

        state_dict_path = f"final_model/{self.best_ckpt}"

        print(f"⌛ Fetching remote artifact at {str(state_dict_path)}")

        local_model_path = mlflow.artifacts.download_artifacts(run_id=self.bean.run_id, artifact_path=str(state_dict_path))
        state_dict = torch.load(local_model_path, map_location="cuda")

        self.model.load(state_dict)
        print(f"✅ Successfully loaded model")

    @staticmethod
    def from_training_bean(bean: Training):
        model_training = ModelTraining(bean.model, bean.dataset.name, bean.x_transform, bean.y_transform,
                                       bean.batch_size, bean.data_amount, bean=bean)
        return model_training


    def prediction_ds_path(self, benchmark_ds_name):
        return f"{self.name}/{benchmark_ds_name}-prediction.zarr"

    def benchmark(self, benchmark_ds_name="re200-sr1etot"):
        dataset = Dataset.get_dataset_or_fail(benchmark_ds_name)
        ds = prepare_dataset(dataset, 1)
        prediction = get_prediction_ds(self, ds)
        s3_access.store_ds(prediction, self.prediction_ds_path(benchmark_ds_name))

    def _test_model(self):
        mse = test_gan(self, self.test_ds)
        print(f"MSE: {mse}")

