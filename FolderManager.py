from pathlib import Path


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from space_exploration.models.GAN3D import GAN3D


class FolderManager:
    channel_coordinates = Path("channel_coordinates")
    _checkpoints = Path("checkpoints")
    _generated_data = Path("generated_data")
    _logs = Path("logs")
    tfrecords = Path("tfrecords")

    @staticmethod
    def checkpoints(model: 'GAN3D'):
        return FolderManager._checkpoints / model.name

    @staticmethod
    def generated_data(model: 'GAN3D'):
        return FolderManager._generated_data / model.name

    @staticmethod
    def logs(model: 'GAN3D'):
        return FolderManager._logs / model.name
