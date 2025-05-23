from sqlalchemy import Integer, Column, Float, ForeignKey
from sqlalchemy.orm import relationship

from space_exploration.beans.alchemy_base import Base


class DatasetStat(Base):
    __tablename__ = 'dataset_stats'
    id = Column(Integer, primary_key=True)
    y_index = Column(Float)
    u_mean = Column(Float)
    v_mean = Column(Float)
    w_mean = Column(Float)
    u_std = Column(Float)
    v_std = Column(Float)
    w_std = Column(Float)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    dataset = relationship("Dataset", back_populates="stats")


