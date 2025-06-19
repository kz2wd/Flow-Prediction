from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar

from space_exploration.dataset import s3_access
from space_exploration.dataset.benchmarks.benchmark_keys import BenchmarkKeys

if TYPE_CHECKING:
    from space_exploration.beans.dataset_bean import Dataset

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans



class Benchmark:
    BENCHMARK_BUCKET = "benchmarks"
    def __init__(self, dataset: 'Dataset', subset_size):
        self.subset_size = subset_size
        self.dataset = dataset
        self.loaded = False
        self.benchmarks = {}


    def load(self):
        self.benchmarks = {}
        for name in BenchmarkKeys:
            df = s3_access.load_df(self.get_benchmark_storage_name(name))
            if df is not None:
                self.benchmarks[name] = df

        self.loaded = len(self.benchmarks) > 0


    def compute(self):

        print("Computing Benchmark Data")

        # ds shape: (Batch, velocity component, x, y, z)
        ds = self.dataset.y * self.dataset.scaling

        y_start = 1 if self.dataset.channel.discard_first_y else 0
        y_dimension = self.dataset.channel.get_simulation_channel().y_dimension
        max_y = ds.shape[3]
        y_dimension = y_dimension[y_start: max_y]
        y_dimension = y_dimension * self.dataset.channel.y_scale_to_y_plus

        def run_benchmark(internal_ds):
            velocity_mean = internal_ds.mean(axis=(0, 2, 4)).compute()  # (3, y)
            velocity_std = internal_ds.std(axis=(2, 4)).mean(axis=0).compute()  # (3, y)
            fluctuation = internal_ds - velocity_mean[None, :, None, :, None]
            squared_velocity_mean = (fluctuation ** 2).mean(axis=(0, 2, 4)).compute()  # (3, y)
            reynolds_uv = (fluctuation[:, 0] * fluctuation[:, 1]).mean(axis=(0, 1, 3)).compute()  # (y,)

            return {
                BenchmarkKeys.VELOCITY_MEAN_ALONG_Y: velocity_mean,
                BenchmarkKeys.VELOCITY_STD_ALONG_Y: velocity_std,
                BenchmarkKeys.FLUCTUATION_ALONG_Y: fluctuation,
                BenchmarkKeys.SQUARED_VELOCITY_MEAN_ALONG_Y: squared_velocity_mean,
                BenchmarkKeys.REYNOLDS_UV: reynolds_uv,
            }

        ds = ds[:self.subset_size, :, :, y_start:, :]

        benchmark_dict = run_benchmark(ds)

        components = ['u', 'v', 'w']
        y_size = len(y_dimension)

        not_aware_benchmarks = []
        channel_aware_benchmarks = [
            BenchmarkKeys.REYNOLDS_UV,
        ]
        component_aware_benchmarks = [
            BenchmarkKeys.VELOCITY_MEAN_ALONG_Y,
            BenchmarkKeys.VELOCITY_STD_ALONG_Y,
            # BenchmarkKeys.FLUCTUATION_ALONG_Y,
            BenchmarkKeys.SQUARED_VELOCITY_MEAN_ALONG_Y,
        ]

        def not_aware(benchmark_name):
            return {
                benchmark_name: benchmark_dict[benchmark_name].flatten(),
                'dataset_id': self.dataset.id,
                'name': str(self.dataset.name),
            }

        def channel_aware(benchmark_name):
            return {
                benchmark_name: benchmark_dict[benchmark_name].flatten(),
                'y_dimension': y_dimension,
                'dataset_id': self.dataset.id,
                'name': str(self.dataset.name),
            }
        def component_aware(benchmark_name):
            return {
                benchmark_name: benchmark_dict[benchmark_name].flatten(),
                'component': np.repeat(components, y_size),
                'y_dimension': np.tile(y_dimension, len(components)),
                'dataset_id': self.dataset.id,
                'name': str(self.dataset.name),
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

        print(f"Saved benchmarks for {self.dataset.name}")


    def get_benchmark_storage_name(self, benchmark_name):
        return f"s3://{self.BENCHMARK_BUCKET}/{self.dataset.name}/{benchmark_name}.parquet"


def compute_pca_coverage(ds, n_components=50):
    N = ds.shape[0]
    spatial_dims = np.prod(ds.shape[2:])
    flattened = ds.reshape((N, 3 * spatial_dims))
    sample = flattened[:500].compute()  # sample subset

    pca = PCA(n_components=n_components)
    pca.fit(sample)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    n_95 = np.searchsorted(cumulative, 0.5) + 1
    return {
        "n_components_95%": n_95,
        "cumulative_variance": cumulative
    }


def compute_state_coverage(ds):
    N = ds.shape[0]
    flattened = ds.reshape((N, -1))
    sample = flattened[:500].compute()  # subset to speed up

    distances = pairwise_distances(sample)
    spread = distances.std()
    mean_dist = distances.mean()

    return {
        "pairwise_std": spread,
        "pairwise_mean": mean_dist,
    }


def prediction_difficulty(ds, n_clusters=10):
    X_wall = ds[:, :, 0, :, :].reshape(ds.shape[0], -1).compute()
    Y_full = ds.reshape(ds.shape[0], -1).compute()

    kmeans = KMeans(n_clusters=n_clusters).fit(X_wall)
    labels = kmeans.labels_

    per_cluster_var = []
    for i in range(n_clusters):
        cluster_points = Y_full[labels == i]
        if len(cluster_points) > 1:
            var = np.var(cluster_points, axis=0).mean()
            per_cluster_var.append(var)

    return {
        "avg_variance_per_cluster": np.mean(per_cluster_var),
        "max_variance_per_cluster": np.max(per_cluster_var),
    }
