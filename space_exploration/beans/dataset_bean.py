import dask.array as da
import numpy as np
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Integer, ForeignKey

from sqlalchemy.orm import relationship

from space_exploration.beans.alchemy_base import Base
from space_exploration.dataset import s3_access
from space_exploration.dataset.analyzer import DatasetAnalyzer
from space_exploration.dataset.dataset_stat import DatasetStats
from space_exploration.dataset.normalize.normalizer_base import NormalizerBase
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
    channel = relationship("Channel")

    def load_s3(self):
        s3_map = s3_access.get_s3_map(f"s3://simulations/{self.s3_storage_name}")
        return da.from_zarr(s3_map)

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

    def get_training_dataset(self, normalizer: NormalizerBase, max_y, size=-1):
        ds = self.load_s3()[:size]
        s3 = S3Dataset(ds, max_y, normalizer)
        return s3

    def get_dataset_analyzer(self):
        return DatasetAnalyzer(self.load_s3(), self.channel.get_simulation_channel())




