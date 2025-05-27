import json
from pathlib import Path
from typing import Type

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from space_exploration.beans.alchemy_base import Base
from space_exploration.beans.channel_bean import Channel
from space_exploration.beans.channel_y_bean import ChannelY
from space_exploration.beans.dataset_bean import Dataset
from space_exploration.beans.dataset_stat_bean import DatasetStat
from space_exploration.beans.prediction_subspace_bean import PredictionSubSpace

creds_path = Path.home() / ".pg_creds.json"
with open(creds_path) as f:
    creds = json.load(f)

url = f"postgresql://{creds['user']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['database']}"

def get_db_url():
    return url

# IMPORT ALL THE BEANS TO MAKE SURE THEY ARE LOADED AT RUNTIME !!! :)
Channel()
ChannelY()
Dataset()
DatasetStat()
PredictionSubSpace()


engine = create_engine(get_db_url())
Session = sessionmaker(bind=engine)

table_init = False

def get_session():
    global table_init
    if not table_init:
        Base.metadata.create_all(engine)
        table_init = True
    return Session()

def get_full_base():
    return Base