import dask.array as da
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Integer, ForeignKey

from sqlalchemy.orm import relationship

from space_exploration.beans.alchemy_base import Base
from space_exploration.dataset import s3_access
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


    def get_training_dataset(self, normalizer: NormalizerBase, max_y, size=-1):
        ds = self.load_s3()[:size]
        s3 = S3Dataset(ds, max_y, normalizer)
        return s3




