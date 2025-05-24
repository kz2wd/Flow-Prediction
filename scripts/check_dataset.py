from typing import TYPE_CHECKING

from space_exploration.dataset import db_access
from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset.normalize.not_normalized import NotNormalized

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

if __name__ == '__main__':
    session: 'Session' = db_access.get_session()
    target_dataset = "paper-validation"

    print(session.query(Dataset).all())

    result: Dataset | None = session.query(Dataset).filter_by(name=target_dataset).first()
    if result is None:
        print("Dataset not found")
        exit(1)

    sim_channel = result.channel.get_simulation_channel()
    print(sim_channel.y_dimension)

    analyzer = result.get_dataset_analyzer()
    print(analyzer.ds.mean().compute())
