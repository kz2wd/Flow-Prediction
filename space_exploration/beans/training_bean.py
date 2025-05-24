from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship

from space_exploration.beans.alchemy_base import Base


class Training(Base):
    __tablename__ = 'trainings'
    id = Column(Integer, primary_key=True)
    prediction_sub_space = relationship("PredictionSubSpace")
    prediction_sub_space_id = Column(Integer, ForeignKey('prediction_subspaces.id'))
    dataset = relationship("Dataset")
    dataset_id = Column(Integer, ForeignKey('datasets.id'))

    # needs the network model too