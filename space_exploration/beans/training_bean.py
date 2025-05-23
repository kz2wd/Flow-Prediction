from sqlalchemy import Column, Integer
from sqlalchemy.orm import relationship

from space_exploration.beans.alchemy_base import Base


class Training(Base):
    __tablename__ = 'trainings'
    id = Column(Integer, primary_key=True)
    prediction_sub_space = relationship("PredictionSubSpace")
