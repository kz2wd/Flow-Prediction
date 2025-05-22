from typing import TYPE_CHECKING

from space_exploration.dataset import db_access
from space_exploration.dataset.dataset_bean import Dataset

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

if __name__ == '__main__':
    session: 'Session' = db_access.get_session()
    result = session.query(Dataset).all()
    if result:
        print(result)
    else:
        print('No dataset')