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
from space_exploration.training.training_benchmark import TrainingBenchmark
from space_exploration.training.training_utils import get_split_datasets


class ModelTraining:
    def __init__(self, model_name, dataset_name, x_transform_name, y_transform_name, batch_size, data_amount=-1, max_epochs=50, saving_freq=3, train_patience=4, name=None, bean=None, profile=False):
        self.profile = profile
        mlflow.set_tracking_uri("http://localhost:5000")
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
            self.name = f"{self.model_name}_{self.dataset_name}_{self.data_amount}_{self.batch_size}"

        # === ARTIFACT MANAGEMENT ===
        self.artifact_dir = Path(FolderManager.artifact_backup_folder(self.model))
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.latest_ckpt = self.artifact_dir / "checkpoint_latest.pt"
        self.best_ckpt = self.artifact_dir / "checkpoint_best.pt"

    def run(self):
        with mlflow.start_run(run_name=self.name):

            run = mlflow.active_run()
            self._record_in_database(run.info.run_id)

            self._prepare_datasets()

            try:
                self._internal_train()
            except KeyboardInterrupt as e:
                print(f"🚩 Stopping training due to user request")
            except Exception as e:
                print(f"Encountered exception while training: {e}")
                import traceback
                traceback.print_exc()


            finally:
                self._upload_best_model()
                self.model.training_end()
                self._clean_training()

    def _prepare_datasets(self):
        y_dim = self.model.prediction_sub_space.y[1]
        ds = self.dataset.get_training_dataset(y_dim, self.x_transform_ref.transformation, self.y_transform_ref.transformation, self.data_amount)
        self.train_ds, self.val_ds, self.test_ds = get_split_datasets(ds, batch_size=4, val_ratio=0.2, test_ratio=0.05,
                                                 device=self.model.device)



    def _clean_training(self):
        print(f"🔄️ Deleting local backups in [{self.artifact_dir}]")
        shutil.rmtree(self.artifact_dir)


    def _upload_best_model(self):
        print("🕓 Uploading best model")
        mlflow.log_artifact(str(self.best_ckpt), artifact_path="final_model")

    def _internal_train(self):

        self.model.prepare_train(self.train_ds, self.val_ds, self.test_ds)

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

            if self.profile:
                log_dir = f"log_dir/{self.name}"
                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA
                        ],
                        schedule=torch.profiler.schedule(wait=0, warmup=5, active=1, repeat=0),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=True
                ) as prof:
                    val_gen_loss = self.model.train_cycle(epoch, start_time, prof)
                    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                    input(f"Done profiling epoch {epoch}, press [Enter] to continue")
            else:
                val_gen_loss = self.model.train_cycle(epoch, start_time)

            if epoch % self.saving_freq == 0:
                self.model.save(epoch, self.latest_ckpt)
                print(f"Saving checkpoint for epoch {epoch}...")

            if val_gen_loss < best_val_loss:
                best_val_loss = val_gen_loss
                patience_counter = 0
                self.model.save(epoch, self.best_ckpt)
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
            name=f"{run_id[:80]}-{self.name}",
            parent=self.bean
        )
        global_session.add(training)
        global_session.commit()

        self.bean = training


    def load_model(self):
        if self.bean is None:
            raise Exception("No existing training record!")

        state_dict_path = "final_model/checkpoint_best.pt"

        print(f"⌛ Fetching remote artifact at {str(state_dict_path)}")

        local_model_path = mlflow.artifacts.download_artifacts(run_id=self.bean.run_id, artifact_path=str(state_dict_path))
        state_dict = torch.load(local_model_path, map_location="cuda")

        self.model.load(state_dict)
        print(f"✅ Successfully loaded model")

    @staticmethod
    def from_training_bean(bean: Training):
        model_training = ModelTraining(bean.model, bean.dataset.name, bean.x_transform, bean.y_transform,
                                       bean.batch_size, bean.data_amount, bean=bean, name=bean.name )
        return model_training


    def prediction_ds_path(self, benchmark_ds_name):
        return f"benchmarks/training-prediction/{self.bean.run_id}/{benchmark_ds_name}-prediction.zarr"

    def get_benchmark(self, benchmark_ds_name="re200-sr1etot"):
        return TrainingBenchmark(self, benchmark_ds_name)

    def change_dataset(self, new_dataset_name, new_data_amount=-1):
        self.dataset_name = new_dataset_name
        self.dataset = Dataset.get_dataset_or_fail(new_dataset_name)
        self.data_amount = new_data_amount
        if self.data_amount == -1:
            self.data_amount = self.dataset.size
