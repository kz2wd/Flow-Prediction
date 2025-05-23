from sqlalchemy import Integer, Column, Float, ForeignKey
from sqlalchemy.orm import relationship

from space_exploration.beans.alchemy_base import Base


class ChannelY(Base):
    __tablename__ = 'channel_y'
    id = Column(Integer, primary_key=True)
    y_index = Column(Integer)
    y_coord = Column(Float)
    channel_id = Column(Integer, ForeignKey('channels.id'))
    channel = relationship("Channel", back_populates="y_dimension")