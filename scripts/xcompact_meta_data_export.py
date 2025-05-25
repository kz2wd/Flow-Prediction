import argparse

from scripts.database_add import add_dataset
from scripts.parser_utils import dir_path
from space_exploration.beans.channel_bean import Channel
from space_exploration.dataset import db_access
from space_exploration.dataset.dataset_stat import DatasetStats
from space_exploration.dataset.s3_access import get_ds

if __name__ == "__main__":
    session = db_access.get_session()

    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-path", required=True, type=dir_path)
    parser.add_argument("--s3-dataset", required=True, type=str)
    parser.add_argument("--dataset-name", required=True, type=str)
    parser.add_argument("--scaling", required=True, type=float)
    parser.add_argument("--existing-channel-name", required=True, type=str)

    args = parser.parse_args()

    channel = Channel.get_channel(session, args.existing_channel_name)
    if channel is None:
        print(f"Channel {args.existing_channel_name} not found ❌")
        print("Available channels:")
        print(*(channel.name for channel in session.query(Channel).all()))
        exit(1)

    ds = get_ds(args.s3_dataset)

    stats = DatasetStats.from_ds(ds)

    add_dataset(
        session=session,
        name=args.dataset_name,
        s3_storage_name=args.s3_dataset,
        scaling=args.scaling,
        channel=channel,
        stats=stats,
    )

    session.commit()

    print(f"Dataset [{args.dataset_name}] exported ✅")
