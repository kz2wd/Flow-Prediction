import numpy as np

from FolderManager import FolderManager
from space_exploration.data_viz import Plotter
from space_exploration.data_viz.PlotData import PlotData
from space_exploration.models.A.A03 import A03
from space_exploration.models.C.C04 import C04
from space_exploration.models.C.C05 import C05
from space_exploration.models.C.C06 import C06
from space_exploration.models.GAN3D import GAN3D
from space_exploration.models.PaperBase import PaperBase
from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace
from space_exploration.simulation_channel.SimulationChannel import SimulationChannel


def test():

    prediction_sub_space = PredictionSubSpace(x_start=0, x_end=64, y_start=0, y_end=64, z_start=0,
                                              z_end=64)
    channel = SimulationChannel(x_length=np.pi, z_length=np.pi / 2, x_resolution=64, z_resolution=64,
                                prediction_sub_space=prediction_sub_space,
                                channel_data_file=FolderManager.tfrecords / "scaling.npz")

    model = GAN3D("torch_test", "ckpt-1", channel)
    # model.test(10)
    # model.benchmark()
    model.train(1, 1, 4, 32)
    # Plotter.plot_mse(model, "mse")


if __name__ == '__main__':
    # test_multiple()
    test()