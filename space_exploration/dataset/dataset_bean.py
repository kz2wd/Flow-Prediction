from dask.array import da
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

from sqlalchemy.orm import relationship

from space_exploration.dataset import s3_access

Base = declarative_base()

class Dataset(Base):
    __tablename__ = 'datasets'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    s3_storage_name = Column(String, unique=True)
    raw_data_normalized = Column(Boolean)
    stats = relationship("DatasetStat", back_populates="dataset", cascade="all, delete-orphan")

    def load_s3(self):
        s3_map = s3_access.get_s3_map(f"s3://simulations/{self.s3_storage_name}")
        return da.from_zarr(s3_map)

    def get_normalized_data(self, ds):
        if self.raw_data_normalized:
            return ds
        # normalize somehow


        return ds


    def get_training_dataset(self, size):
        ds = self.load_s3()[:size]
        normalized = self.get_normalized_data(ds)




class DatasetStat(Base):
    __tablename__ = 'dataset_stats'
    id = Column(Integer, primary_key=True)
    y_index = Column(Float)
    y_coord = Column(Float)
    u_mean = Column(Float)
    v_mean = Column(Float)
    w_mean = Column(Float)
    u_std = Column(Float)
    v_std = Column(Float)
    w_std = Column(Float)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    dataset = relationship("Dataset", back_populates="stats")



