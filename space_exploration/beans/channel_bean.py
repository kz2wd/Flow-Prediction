import numpy as np
from sqlalchemy import Column, Integer, String, Float, Boolean
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
    discard_first_y = Column(Boolean, default=False)  # When value too close to wall used, 'breaks' graphs

    def get_simulation_channel(self) -> SimulationChannel:
        y_dimension = np.array([channel_y.y_coord for channel_y in self.y_dimension])
        channel = SimulationChannel(self.x_length, self.x_resolution, self.z_length, self.z_resolution, y_dimension,
                                    self.y_scale_to_y_plus)
        return channel

    @staticmethod
    def get_channel(name):
        from space_exploration.dataset.db_access import global_session
        return global_session.query(Channel).filter_by(name=name).first()

    @staticmethod
    def get_channel_or_fail(name):
        from space_exploration.dataset.db_access import global_session
        channel = Channel.get_channel(global_session, name)
        if channel is None:
            print(f"Channel {name} not found ❌")
            print("Available channels:")
            print(*(channel.name for channel in global_session.query(Channel).all()))
            exit(1)
        return channel
