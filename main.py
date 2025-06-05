from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset import db_access
from space_exploration.dataset.normalize.not_normalized import NotNormalized
from space_exploration.models.implementations.A import A
from space_exploration.training import train_gan, get_split_datasets, test_gan


def test():
    session = db_access.get_session()
    dataset = Dataset.get_dataset_or_fail(session, "re200-sr001etot")
    model = A("A-test")
    y_dim = model.prediction_sub_space.y[1]
    model_ds = dataset.get_training_dataset(NotNormalized(), y_dim)

    train_ds, val_ds, test_ds = get_split_datasets(model_ds, batch_size=8, val_ratio=0.1, test_ratio=0.1, device=model.device)

    train_gan(model, train_ds, val_ds)
    test_gan(model, test_ds)

if __name__ == '__main__':
    test()