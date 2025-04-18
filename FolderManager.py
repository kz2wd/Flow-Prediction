from pathlib import Path


from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from space_exploration.models.GAN3D import GAN3D


class FolderManager:
    channel_coordinates = Path("channel_coordinates")
    _checkpoints = Path("checkpoints")
    _generated_data = Path("generated_data")
    _logs = Path("logs")
    tfrecords = Path("tfrecords")

    @staticmethod
    def init(model: 'GAN3D'):
        FolderManager.checkpoints(model).mkdir(parents=True, exist_ok=True)
        FolderManager.generated_data(model).mkdir(parents=True, exist_ok=True)
        FolderManager.logs(model).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def checkpoints(model: 'GAN3D'):
        return FolderManager._checkpoints / (model if isinstance(model, str) else model.name)

    @staticmethod
    def generated_data(model: 'GAN3D'):
        return FolderManager._generated_data / (model if isinstance(model, str) else model.name)

    @staticmethod
    def logs(model: 'GAN3D'):
        return FolderManager._logs / (model if isinstance(model, str) else model.name)

    @staticmethod
    def benchmark_file(model: 'GAN3D'):
        return FolderManager.generated_data(model) / "benchmark.hdf5"

    @staticmethod
    def predictions_file(model: 'GAN3D'):
        return FolderManager.generated_data(model) / "predictions.hdf5"