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

    dataset = result.get_training_dataset(NotNormalized(), 64, 10)
    print(dataset.ds.mean().compute())
