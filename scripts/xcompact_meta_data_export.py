import argparse

from space_exploration.beans.database_add import add_dataset
from scripts.parser_utils import dir_path
from space_exploration.beans.channel_bean import Channel
from space_exploration.dataset.db_access import global_session
from space_exploration.dataset.s3_access import get_ds

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-path", required=True, type=dir_path)
    parser.add_argument("--dataset-name", required=True, type=str)
    parser.add_argument("--scaling", required=True, type=float)
    parser.add_argument("--existing-channel-name", required=True, type=str)

    args = parser.parse_args()

    channel = Channel.get_channel_or_fail(args.existing_channel_name)

    ds = get_ds(args.s3_dataset)
    add_dataset(args.dataset_name, args.scaling, channel)

    global_session.commit()

    print(f"Dataset [{args.dataset_name}] exported ✅")
