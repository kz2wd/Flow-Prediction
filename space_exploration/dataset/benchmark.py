from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar

from space_exploration.dataset import s3_access

if TYPE_CHECKING:
    from space_exploration.beans.dataset_bean import Dataset

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans


class Benchmark:
    BENCHMARK_BUCKET = "benchmarks"
    BASE_DF = "base"
    CHANNEL_DF = "channel"
    COMPONENT_DF = "component"
    def __init__(self, dataset: 'Dataset'):
        self.dataset = dataset


    def load(self):
        self.base_df = s3_access.get_ds(self.get_benchmark_storage_name(self.BASE_DF))
        self.channel_df = s3_access.get_ds(self.get_benchmark_storage_name(self.CHANNEL_DF))
        self.component_df = s3_access.get_ds(self.get_benchmark_storage_name(self.COMPONENT_DF))

    def compute(self):
        print("Computing Benchmark Data")

        # ds shape: (Batch, velocity component, x, y, z)
        ds = self.dataset.load_s3() * self.dataset.scaling

        y_start = 1 if self.dataset.channel.discard_first_y else 0
        y_dimension = self.dataset.channel.get_simulation_channel().y_dimension
        max_y = ds.shape[3]
        y_dimension = y_dimension[y_start: max_y]
        y_dimension = y_dimension * self.dataset.channel.y_scale_to_y_plus

        ds = ds[:, :, :, y_start:, :]


        with ProgressBar():
            velocity_mean = ds.mean(axis=(0, 2, 4)).compute()  # (3, y)
            velocity_std = ds.std(axis=(2, 4)).mean(axis=0).compute()  # (3, y)
            fluctuation = ds - velocity_mean[None, :, None, :, None]
            squared_velocity_mean = (fluctuation ** 2).mean(axis=(0, 2, 4)).compute()  # (3, y)
            reynolds_uv = (fluctuation[:, 0] * fluctuation[:, 1]).mean(axis=(0, 1, 3)).compute()  # (y,)

        components = ['u', 'v', 'w']
        y_size = len(y_dimension)

        base_dict = {
            **compute_pca_coverage(ds),
            **prediction_difficulty(ds),
            'dataset_id': self.dataset.id,
            'name': str(self.dataset.name),
        }

        channel_dict = {
            'reynolds_uv': reynolds_uv.flatten(),
            'y_dimension': y_dimension,
            'dataset_id': self.dataset.id,
            'name': str(self.dataset.name),
        }

        component_dict = {
            'component': np.repeat(components, y_size),

            'velocity_mean': velocity_mean.flatten(),
            'velocity_std': velocity_std.flatten(),
            'squared_velocity_mean': squared_velocity_mean.flatten(),

            'dataset_id': self.dataset.id,
            'name': str(self.dataset.name),
        }

        #** compute_state_coverage(ds),

        s3_access.store_df(pd.DataFrame(base_dict), self.get_benchmark_storage_name(self.BASE_DF))
        s3_access.store_df(pd.DataFrame(channel_dict), self.get_benchmark_storage_name(self.CHANNEL_DF))
        s3_access.store_df(pd.DataFrame(component_dict), self.get_benchmark_storage_name(self.COMPONENT_DF))
        print(f"Saved benchmark for {self.dataset.name}")


    def get_benchmark_storage_name(self, df_type):
        return f"s3://{self.BENCHMARK_BUCKET}/{self.dataset.name}/{df_type}.parquet"


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


def benchmark_dataset(ds):
    print("Im old...")
