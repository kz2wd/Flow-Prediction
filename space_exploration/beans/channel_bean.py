import numpy as np
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import relationship

from space_exploration.beans.alchemy_base import Base
from space_exploration.simulation_channel.SimulationChannel import SimulationChannel


class Channel(Base):
    __tablename__ = 'channels'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    x_resolution = Column(Integer)
    z_resolution = Column(Integer)
    x_length = Column(Float)
    z_length = Column(Float)
    y_scale_to_y_plus = Column(Float)
    y_dimension = relationship("ChannelY", back_populates="channel", cascade="all, delete-orphan")

    def get_simulation_channel(self):
        y_dimension = np.array([channel_y.y_coord for channel_y in self.y_dimension])
        channel = SimulationChannel(self.x_length, self.x_resolution, self.z_length, self.z_resolution, y_dimension,
                                    self.y_scale_to_y_plus)
        return channel

    @staticmethod
    def get_channel(session, name):
        return session.query(Channel).filter_by(name=name).first()

    @staticmethod
    def get_channel_or_fail(session, name):
        channel = Channel.get_channel(session, name)
        if channel is None:
            print(f"Channel {name} not found ❌")
            print("Available channels:")
            print(*(channel.name for channel in session.query(Channel).all()))
            exit(1)
        return channel
