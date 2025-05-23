from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import relationship

from space_exploration.beans.alchemy_base import Base


class Channel(Base):
    __tablename__ = 'channels'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    x_resolution = Column(Integer)
    z_resolution = Column(Integer)
    x_length = Column(Float)
    z_length = Column(Float)
    y_dimension = relationship("ChannelY", back_populates="channel", cascade="all, delete-orphan")
