from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

from sqlalchemy.orm import relationship

Base = declarative_base()

class Dataset(Base):
    __tablename__ = 'datasets'

    id = Column(String, primary_key=True)
    name = Column(String, unique=True)
    s3_storage_name = Column(String, unique=True)

    stats = relationship("DatasetStat", back_populates="dataset", cascade="all, delete-orphan")


class DatasetStat(Base):
    __tablename__ = 'dataset_stats'
    id = Column(String, primary_key=True)
    y_index = Column(Float)
    y_coord = Column(Float)
    velocity_mean = Column(Float)
    velocity_std = Column(Float)

    dataset = relationship("Dataset", back_populates="stats")



