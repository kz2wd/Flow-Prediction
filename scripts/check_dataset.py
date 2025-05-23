from typing import TYPE_CHECKING

from space_exploration.dataset import db_access
from space_exploration.beans.dataset_bean import Dataset

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

if __name__ == '__main__':
    session: 'Session' = db_access.get_session()
    target_dataset = "paper-validation"

    result = session.query(Dataset).filter( Dataset.name == target_dataset).first()
    if result is None:
        print("Dataset not found")
        exit(1)


    result.get_training_dataset()
