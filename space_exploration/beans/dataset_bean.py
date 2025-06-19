import functools
from typing import Type

import numpy as np
import vtk
from sqlalchemy import Column, String, Float, Integer, ForeignKey
from sqlalchemy.orm import relationship, Mapped
from vtk.util import numpy_support

from space_exploration.FolderManager import FolderManager
from space_exploration.beans.alchemy_base import Base
from space_exploration.beans.channel_bean import Channel
from space_exploration.dataset import s3_access
from space_exploration.dataset.analyzer import DatasetAnalyzer
from space_exploration.dataset.benchmark import Benchmark
from space_exploration.dataset.dataset_stat import DatasetStats
from space_exploration.dataset.db_access import global_session
from space_exploration.dataset.transforms.general.default_unchanged import DefaultUnchanged
from space_exploration.dataset.s3_dataset import S3Dataset


class Dataset(Base):
    __tablename__ = 'datasets'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    scaling = Column(Float)
    stats = relationship("DatasetStat", back_populates="dataset", cascade="all, delete-orphan")
    channel_id = Column(Integer, ForeignKey('channels.id'))
    channel: Mapped[Channel] = relationship("Channel")

    @property
    def ds_s3_storage(self):
        return f"simulations/{self.name}.zarr"

    @functools.cached_property
    def x(self):
        return s3_access.get_x(self.ds_s3_storage)

    @functools.cached_property
    def y(self):
        return s3_access.get_y(self.ds_s3_storage)

    def reload_ds(self):
        self.__dict__.pop('x', None)
        self.__dict__.pop('y', None)

    def get_stats(self):
        u_means = []
        v_means = []
        w_means = []
        u_stds = []
        v_stds = []
        w_stds = []
        for stat in self.stats:
            u_means.append(stat.u_mean)
            v_means.append(stat.v_mean)
            w_means.append(stat.w_mean)
            u_stds.append(stat.u_std)
            v_stds.append(stat.v_std)
            w_stds.append(stat.w_std)
        return DatasetStats(np.array(u_means), np.array(v_means), np.array(w_means), np.array(u_stds), np.array(v_stds), np.array(w_stds))

    @property
    def size(self):
        return self.x.shape[0]

    def get_training_dataset(self, max_y, XTransform: Type = DefaultUnchanged, YTransform: Type = DefaultUnchanged, size=-1):
        s3 = S3Dataset(self, self.x[:size], self.y[:size], max_y, XTransform, YTransform)
        return s3

    def get_dataset_analyzer(self):
        return DatasetAnalyzer(self.y, self.scaling, self.channel.get_simulation_channel())

    @functools.cached_property
    def benchmark(self):
        benchmark = Benchmark(self)
        benchmark.load()
        return benchmark

    def reload_benchmark(self):
        self.reload_ds()
        self.__dict__.pop('benchmark', None)

    def export_frame_vts(self, frame, array_name="velocity_target"):

        target = self.y[frame].compute()

        y_len = target.shape[2]

        target = np.transpose(target, (1, 2, 3, 0)).reshape(-1, 3)

        structured_grid = vtk.vtkStructuredGrid()
        points = vtk.vtkPoints()

        channel = self.channel.get_simulation_channel()



        for k in range(len(channel.z_dimension)):
            for j in range(y_len):
                for i in range(len(channel.x_dimension)):
                    points.InsertNextPoint(channel.x_dimension[i], channel.y_dimension[j],
                                           channel.z_dimension[k])

        structured_grid.SetPoints(points)
        structured_grid.SetDimensions(len(channel.x_dimension), y_len, len(channel.z_dimension))

        velocity_array = numpy_support.numpy_to_vtk(num_array=target, deep=True,
                                                    array_type=vtk.VTK_FLOAT)
        velocity_array.SetName(array_name)

        structured_grid.GetPointData().AddArray(velocity_array)

        writer = vtk.vtkXMLStructuredGridWriter()
        filename = FolderManager.vts_frames_folder(self) / f"frame_{frame}.vts"
        writer.SetFileName(filename)
        writer.SetInputData(structured_grid)
        writer.Write()
        print(f"Wrote frame {frame} to {filename}")

    @staticmethod
    def get_dataset_or_fail(name):
        result: Dataset | None = global_session.query(Dataset).filter_by(name=name).first()
        if result is None:
            print(f"Dataset [{name}] not found ❌")
            print("Available datasets:")
            print(*(dataset.name for dataset in global_session.query(Dataset).all()))
            raise Exception(f"Dataset [{name}] not found <UNK>")
        return result

