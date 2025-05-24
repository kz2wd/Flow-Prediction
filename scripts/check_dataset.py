from argparse import ArgumentParser
from typing import TYPE_CHECKING

from space_exploration.dataset import db_access
from space_exploration.beans.dataset_bean import Dataset

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset", required=True)
    args = arg_parser.parse_args()

    session: 'Session' = db_access.get_session()
    target_dataset = args.dataset

    print()

    result: Dataset | None = session.query(Dataset).filter_by(name=target_dataset).first()
    if result is None:
        print(f"Dataset [{target_dataset}] not found")
        print("Available datasets:")
        print(*(dataset.name for dataset in session.query(Dataset).all()))
        exit(1)

    analyzer = result.get_dataset_analyzer()
    analyzer.plot_u_velo_along_y(f"{target_dataset}-velocities.png")
