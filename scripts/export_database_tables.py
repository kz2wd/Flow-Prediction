
from sqlalchemy import create_engine

from space_exploration.beans.alchemy_base import Base
from space_exploration.beans.channel_bean import Channel
from space_exploration.beans.channel_y_bean import ChannelY
from space_exploration.beans.dataset_bean import Dataset
from space_exploration.beans.dataset_stat_bean import DatasetStat
from space_exploration.beans.prediction_subspace_bean import PredictionSubSpace
from space_exploration.dataset import db_access


if __name__ == "__main__":
    # Beans
    Channel()
    ChannelY()
    Dataset()
    DatasetStat()
    PredictionSubSpace()

    engine = create_engine(db_access.get_db_url())
    Base.metadata.create_all(engine)