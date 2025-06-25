from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pandas as pd

from space_exploration.beans.database_add import add_dataset
from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset import s3_access
from space_exploration.dataset.db_access import global_session
from space_exploration.dataset.transforms.AllTransforms import TransformationReferences

from space_exploration.training.training_benchmark_keys import TrainingBenchmarkKeys
from space_exploration.training.training_utils import prepare_dataset, get_prediction_ds

if TYPE_CHECKING:
    from space_exploration.training.training import ModelTraining


class TrainingBenchmark:
    BENCHMARK_BUCKET = "benchmarks"
    def __init__(self, training: 'ModelTraining', benchmark_ds_name):
        self.benchmark_ds_name = benchmark_ds_name
        self.training = training
        self.loaded = False
        self.benchmarks = {}


    def load(self):
        self.benchmarks = {}
        for name in TrainingBenchmarkKeys:
            df = s3_access.load_df(self.get_benchmark_storage_name(name))
            if df is not None:
                self.benchmarks[name] = df

        self.loaded = len(self.benchmarks) > 0


    def compute(self):

        print("🕓 Computing predictions...")

        dataset = Dataset.get_dataset_or_fail(self.benchmark_ds_name)
        ds = dataset.get_training_dataset(
            self.training.model.prediction_sub_space.y[1],
            self.training.x_transform_ref.transformation,
            self.training.y_transform_ref.transformation,
            250
        )
        ds_loader = prepare_dataset(ds, 1)
        prediction = get_prediction_ds(self.training.model, ds_loader)


        print("Computing Benchmark Data")

        y_start = 1 if self.training.dataset.channel.discard_first_y else 0
        y_dimension = self.training.dataset.channel.get_simulation_channel().y_dimension
        max_y = prediction.shape[3]
        y_dimension = y_dimension[y_start: max_y]
        y_dimension = y_dimension * self.training.dataset.channel.y_scale_to_y_plus

        target = da.concatenate([y.cpu().numpy() for (x, y) in iter(ds_loader)], axis=0)

        mse = ((target - prediction) ** 2).mean(axis=(0, 2, 4))

        benchmark_dict = {
            TrainingBenchmarkKeys.PAPER_LIKE_MSE_ALONG_Y: mse,
        }

        components = ['u', 'v', 'w']
        y_size = len(y_dimension)

        not_aware_benchmarks = [
        ]
        channel_aware_benchmarks = [
        ]
        component_aware_benchmarks = [
            TrainingBenchmarkKeys.PAPER_LIKE_MSE_ALONG_Y,
        ]

        def not_aware(benchmark_name):
            return {
                benchmark_name: benchmark_dict[benchmark_name].flatten(),
                'training_id': self.training.bean.id,
                'name': str(self.training.name),
            }

        def channel_aware(benchmark_name):
            return {
                benchmark_name: benchmark_dict[benchmark_name].flatten(),
                'y_dimension': y_dimension,
                'training_id': self.training.bean.id,
                'name': str(self.training.name),
            }
        def component_aware(benchmark_name):
            return {
                benchmark_name: benchmark_dict[benchmark_name].flatten(),
                'component': np.repeat(components, y_size),
                'y_dimension': np.tile(y_dimension, len(components)),
                'training_id': self.training.bean.id,
                'name': str(self.training.name),
            }

        treatments = [
            (not_aware_benchmarks, not_aware),
            (channel_aware_benchmarks, channel_aware),
            (component_aware_benchmarks, component_aware),
        ]

        for benchmark_set, treatment in treatments:
            for benchmark_name in benchmark_set:
                print(f"Saving {benchmark_name}")
                s3_access.store_df(pd.DataFrame(treatment(benchmark_name)), self.get_benchmark_storage_name(benchmark_name))

        # s3_access.store_ds(prediction[0], )

        print(f"Generating prediction dataset benchmark")


        dataset_name = f"prediction-{self.training.bean.run_id}"
        try:
            dataset = add_dataset(
                name=dataset_name,
                channel=self.training.bean.dataset.channel,
                scaling=self.training.bean.dataset.scaling,
                from_dataset=dataset,
                from_training=self.training.bean,
            )
            global_session.commit()
        except Exception:
            global_session.rollback()
            dataset = Dataset.get_dataset_or_fail(dataset_name)

        dataset_benchmark = dataset.benchmark

        real_scale_prediction = self.training.y_transform_ref.transformation().from_training(prediction)

        dataset_benchmark._compute_intern(real_scale_prediction, dataset.channel)

        print(f"Saved benchmarks for {self.training.name}")


    def get_benchmark_storage_name(self, benchmark_name):
        return f"s3://{self.BENCHMARK_BUCKET}/predictions/{self.training.bean.run_id}/{benchmark_name}.parquet"
