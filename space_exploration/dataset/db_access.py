import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from space_exploration.beans.alchemy_base import Base

creds_path = Path.home() / ".pg_creds.json"
with open(creds_path) as f:
    creds = json.load(f)

url = f"postgresql://{creds['user']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['database']}"

def get_db_url():
    return url

engine = create_engine(get_db_url())
Session = sessionmaker(bind=engine)

table_init = False

def get_session():
    global table_init
    if not table_init:
        Base.metadata.create_all(engine)
        table_init = True
    return Session()