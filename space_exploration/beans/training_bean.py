from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship

from space_exploration.beans.alchemy_base import Base


class Training(Base):
    __tablename__ = 'trainings'
    id = Column(Integer, primary_key=True)
    dataset = relationship("Dataset")
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    data_amount = Column(Integer)
    model = Column(String)
    x_transform = Column(String)
    y_transform = Column(String)
    run_id = Column(String)


