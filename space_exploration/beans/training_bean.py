from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship

from space_exploration.beans.alchemy_base import Base


class Training(Base):
    __tablename__ = 'trainings'
    id = Column(Integer, primary_key=True)
    dataset = relationship("Dataset")
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    data_amount = Column(Integer)
    batch_size = Column(Integer)
    model = Column(String)
    x_transform = Column(String)
    y_transform = Column(String)
    run_id = Column(String)

    @staticmethod
    def get_training_or_fail(run_id):
        from space_exploration.dataset.db_access import global_session
        result: Training | None = global_session.query(Training).filter_by(run_id=run_id).first()
        if result is None:
            print(f"Training [{run_id}] not found ❌")
            print("Available Training:")
            print(*(training.run_id for training in global_session.query(Training).all()))
            raise Exception(f"Training [{run_id}] not found <UNK>")
        return result
