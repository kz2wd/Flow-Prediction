from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset import db_access
from space_exploration.dataset.normalize.not_normalized import NotNormalized
from space_exploration.dataset.s3_dataset import S3Dataset
from space_exploration.models.implementations.A import A


def test():

    # model = A("A-test", "")
    # model.train(1, 1, 4, 100)
    session = db_access.get_session()
    dataset = Dataset.get_dataset_or_fail(session, "paper-validation")
    training_ds = dataset.get_training_dataset(NotNormalized(), 64)
    training_ds

if __name__ == '__main__':
    test()