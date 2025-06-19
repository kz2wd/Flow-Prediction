from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset import db_access
from space_exploration.dataset.transforms.AllTransforms import TransformationReferences
from space_exploration.models.AllModels import ModelReferences
from space_exploration.training_utils import train_gan, get_split_datasets


def launch_training(model_ref, dataset_name, x_transform_ref, y_transform_ref):
    dataset = Dataset.get_dataset_or_fail(dataset_name)

    model = model_ref.model()
    y_dim = model.prediction_sub_space.y[1]

    model_ds = dataset.get_training_dataset(y_dim, x_transform_ref.transformation, y_transform_ref.transformation)

    train_ds, val_ds, _ = get_split_datasets(model_ds, batch_size=4, val_ratio=0.1, test_ratio=0.0,
                                                   device=model.device)

    train_gan(model, train_ds, val_ds)


    # test_gan(model, test_ds)


def test():
    launch_training(ModelReferences.A, "re200-sr005etot", TransformationReferences.COMPONENT_NORMALIZE, TransformationReferences.Y_ALONG_COMPONENT_NORMALIZE)


if __name__ == '__main__':
    test()