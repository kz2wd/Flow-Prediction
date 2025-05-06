import h5py
import numpy as np

from FolderManager import FolderManager
from space_exploration.models.GAN3D import GAN3D
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

def fix_dataset():
    with h5py.File(FolderManager.dataset / "test_fixed.hdf5", 'r') as f:
        X = f['x'][...]
        Y = f['y'][...]
        # X = np.squeeze(f['x'], axis=1)
        # Y = np.squeeze(f['y'], axis=1)
        print("X shape: ", X.shape)
        print("Y shape: ", Y.shape)

        input("Enter to continue")
        with h5py.File(FolderManager.dataset / "test_fixed.hdf5", 'w') as fixed:
            fixed.create_dataset('x', data=X)
            fixed.create_dataset('y', data=Y)

if __name__ == '__main__':
    # test_multiple()
    test()
    # fix_dataset()