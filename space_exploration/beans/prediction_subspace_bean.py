from sqlalchemy import Column, Integer, String

from space_exploration.beans.alchemy_base import Base


class PredictionSubSpace(Base):
    __tablename__ = 'prediction_subspaces'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    x_start = Column(Integer)
    x_end = Column(Integer)
    y_start = Column(Integer)
    y_end = Column(Integer)
    z_start = Column(Integer)
    z_end = Column(Integer)