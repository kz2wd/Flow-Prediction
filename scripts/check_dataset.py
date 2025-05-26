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

    dataset = Dataset.get_dataset_or_fail(session, target_dataset)

    analyzer = dataset.get_dataset_analyzer()
    # analyzer.plot_u_velo_along_y(f"{target_dataset}-velocities.png")
    # analyzer.plot_stds(f"{target_dataset}-stds.png")
    analyzer.plot_velocity_fluctuation(f"{target_dataset}-fluctuations.png")