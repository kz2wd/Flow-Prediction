import argparse
import os
from pathlib import Path

import numpy as np

from scripts.database_add import add_dataset
from scripts.parser_utils import dir_path
from space_exploration.beans.channel_bean import Channel
from space_exploration.dataset import db_access

if __name__ == "__main__":
    session = db_access.get_session()

    parser = argparse.ArgumentParser()
    parser.add_argument("simulation_path", metavar="simulation-path", required=True, type=dir_path)
    parser.add_argument("s3_dataset", metavar="s3-dataset", required=True, type=str)
    parser.add_argument("dataset_name", metavar="dataset-name", required=True, type=str)
    parser.add_argument("scaling", required=False, const=1.0, type=float)
    parser.add_argument("existing_channel_name", metavar="existing-channel-name", required=True, type=str)

    args = parser.parse_args()


    # ypi.dat -> y_dimension
    # stats -> compute from ds
    stats = None


    channel = Channel.get_channel(session, args.existing_channel_name)
    if channel is None:
        print("Channel not found")
        exit(1)

    add_dataset(
        session=session,
        name=args.dataset_name,
        s3_storage_name=args.s3_dataset,
        scaling=args.scaling,
        channel=channel,
        stats=stats,
    )

    session.commit()
