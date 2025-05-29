from argparse import ArgumentParser
from typing import TYPE_CHECKING

from space_exploration.beans.channel_bean import Channel
from space_exploration.dataset import db_access
from space_exploration.beans.dataset_bean import Dataset

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--channel", required=True)
    args = arg_parser.parse_args()

    session: 'Session' = db_access.get_session()
    target_channel = args.channel

    channel = Channel.get_channel_or_fail(session, target_channel)

    print(*channel.get_simulation_channel().y_dimension, sep='\n')
