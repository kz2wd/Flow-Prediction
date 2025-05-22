import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

creds_path = Path.home() / ".pg_creds.json"
with open(creds_path) as f:
    creds = json.load(f)

url = f"postgresql://{creds['user']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['database']}"

def get_db_url():
    return url

engine = create_engine(get_db_url())
Session = sessionmaker(bind=engine)


def get_session():
    return Session()