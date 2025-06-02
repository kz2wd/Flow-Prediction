import functools
from functools import lru_cache, cache

import dask.array as da
import numpy as np
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Integer, ForeignKey

from sqlalchemy.orm import relationship, Mapped

from space_exploration.beans.alchemy_base import Base
from space_exploration.beans.channel_bean import Channel
from space_exploration.dataset import s3_access
from space_exploration.dataset.analyzer import DatasetAnalyzer
from space_exploration.dataset.benchmark import benchmark_dataset, Benchmark
from space_exploration.dataset.dataset_stat import DatasetStats
from space_exploration.dataset.normalize.normalizer_base import NormalizerBase
from space_exploration.dataset.s3_access import load_df, exist
from space_exploration.dataset.s3_dataset import S3Dataset
from space_exploration.simulation_channel.SimulationChannel import SimulationChannel



class Dataset(Base):
    __tablename__ = 'datasets'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    s3_storage_name = Column(String, unique=True)
    scaling = Column(Float)
    stats = relationship("DatasetStat", back_populates="dataset", cascade="all, delete-orphan")
    channel_id = Column(Integer, ForeignKey('channels.id'))
    channel: Mapped[Channel] = relationship("Channel")

    def load_s3(self):
        return s3_access.get_ds(self.s3_storage_name)

    def get_stats(self):
        u_means = []
        v_means = []
        w_means = []
        u_stds = []
        v_stds = []
        w_stds = []
        for stat in self.stats:
            u_means.append(stat.u_mean)
            v_means.append(stat.v_mean)
            w_means.append(stat.w_mean)
            u_stds.append(stat.u_std)
            v_stds.append(stat.v_std)
            w_stds.append(stat.w_std)
        return DatasetStats(np.array(u_means), np.array(v_means), np.array(w_means), np.array(u_stds), np.array(v_stds), np.array(w_stds))

    @functools.cached_property
    def size(self):
        return self.load_s3().shape[0]

    def create_benchmark(self):
        benchmark_dataset(self)

    def get_training_dataset(self, normalizer: NormalizerBase, max_y, size=-1):
        ds = self.load_s3()[:size]
        s3 = S3Dataset(ds, max_y, normalizer)
        return s3

    def get_dataset_analyzer(self):
        return DatasetAnalyzer(self.load_s3(), self.scaling, self.channel.get_simulation_channel())

    @functools.cached_property
    def benchmark(self):
        return Benchmark(self)

    def reload_benchmark(self):
        self.__dict__.pop('benchmark', None)

    @staticmethod
    def get_dataset_or_fail(session, name):
        result: Dataset | None = session.query(Dataset).filter_by(name=name).first()
        if result is None:
            print(f"Dataset [{name}] not found ❌")
            print("Available datasets:")
            print(*(dataset.name for dataset in session.query(Dataset).all()))
            exit(1)
        return result

