from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from space_exploration.models.GAN3D import GAN3D


class FolderManager:
    FOLDER_PREFIX_FILE = Path("PREFIX_LOCATION.txt")
    FOLDER_PREFIX = ""
    if FOLDER_PREFIX_FILE.is_file():
        FOLDER_PREFIX = Path(FOLDER_PREFIX_FILE).read_text().strip("\n")
    mlflow_tracking_uri = Path("mlruns")
    channel_coordinates = Path(FOLDER_PREFIX) / "channel_coordinates"
    _checkpoints = Path(FOLDER_PREFIX) / "checkpoints"
    _generated_data = Path(FOLDER_PREFIX) / "generated_data"
    _logs = Path(FOLDER_PREFIX) / "logs"
    tfrecords = Path(FOLDER_PREFIX) / "tfrecords"
    dataset = Path(FOLDER_PREFIX) / "dataset"

    @staticmethod
    def init(model: 'GAN3D'):
        FolderManager.checkpoints(model).mkdir(parents=True, exist_ok=True)
        FolderManager.generated_data(model).mkdir(parents=True, exist_ok=True)
        FolderManager.logs(model).mkdir(parents=True, exist_ok=True)
        FolderManager.dataset.mkdir(parents=True, exist_ok=True)

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