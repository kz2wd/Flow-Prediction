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
    y_dimension = relationship("ChannelY", back_populates="channel", cascade="all, delete-orphan")
    # datasets = relationship("Dataset", back_populates="channel")

    # TODO
    def get_simulation_channel(self):
        channel = SimulationChannel(self.x_length, self.x_resolution, self.z_length, self.z_resolution)

    @staticmethod
    def get_channel(session, name):
        return session.query(Channel).filter_by(name=name).first()