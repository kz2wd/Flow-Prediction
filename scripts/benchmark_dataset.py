from argparse import ArgumentParser
from typing import TYPE_CHECKING

from space_exploration.dataset import db_access
from space_exploration.beans.dataset_bean import Dataset

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from dask.distributed import Client


if __name__ == '__main__':
    # client = Client("tcp://127.0.0.1:8786")
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset", required=True)
    args = arg_parser.parse_args()

    target_dataset = args.dataset

    dataset = Dataset.get_dataset_or_fail(target_dataset)

    dataset.benchmark.compute()
