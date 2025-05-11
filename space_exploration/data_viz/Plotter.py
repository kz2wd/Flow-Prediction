from typing import TYPE_CHECKING

import h5py
import matplotlib.pyplot as plt
import numpy as np

from FolderManager import FolderManager
from space_exploration.data_viz.PlotData import get_benchmarks, PlotData


if TYPE_CHECKING:
    from space_exploration.models.GAN3D import GAN3D

def plot_mse(source_model: 'GAN3D', fig_name):
    u_mse, v_mse, w_mse = get_benchmarks(source_model,
                                         [PlotData.u_mse_y_wise, PlotData.v_mse_y_wise, PlotData.w_mse_y_wise])


    plt.semilogx(source_model.channel.y_channel[1:], u_mse[1:], label='u')
    plt.semilogx(source_model.channel.y_channel[1:], v_mse[1:], label='v')
    plt.semilogx(source_model.channel.y_channel[1:], w_mse[1:], label='w')
    plt.semilogx([15, 30, 50, 100], [0.015, 0.065, 0.195, 0.53], label='Paper u', linestyle=':', marker='D')
    plt.semilogx([15, 30, 50, 100], [0.02, 0.105, 0.30, 0.69], label='Paper v', linestyle=':', marker='D')
    plt.semilogx([15, 30, 50, 100], [0.02, 0.085, 0.28, 0.69], label='Paper w', linestyle=':', marker='D')
    plt.title("MSE of the various velocity components")
    plt.xlabel("y+")
    plt.ylabel("MSE")
    plt.grid()
    plt.grid(which='minor', linestyle='--')
    plt.legend()
    plt.savefig(FolderManager.generated_data(source_model) / fig_name)


def plot_contours(source_model, fig_name):
    target_sample = 0
    with h5py.File(FolderManager.predictions_file(source_model), 'r') as f:
        y_target_normalized = f['y_target'][target_sample, ...]
        y_predict_normalized = f['y_pred'][target_sample, ...]
    #
    plt.clf()
    rows = 2
    cols = 3
    ratio = 0.5
    inches_per_pt = 1.0 / 72.27
    fig_width_pt = 2000
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width * rows / cols * ratio
    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)

    channel_z = source_model.channel.z_dimension
    channel_x = source_model.channel.x_dimension

    axs[0, 0].contourf(channel_x, channel_z, y_target_normalized[:, 22, :, 0].T, vmin=-3, vmax=3,
                       cmap='RdBu_r')
    axs[1, 0].contourf(channel_x, channel_z, y_predict_normalized[:, 22, :, 0].T, vmin=-3, vmax=3,
                       cmap='RdBu_r')
    axs[0, 1].contourf(channel_x, channel_z, y_target_normalized[:, 22, :, 1].T, vmin=-3, vmax=3,
                       cmap='PuOr_r')
    axs[1, 1].contourf(channel_x, channel_z, y_predict_normalized[:, 22, :, 1].T, vmin=-3, vmax=3,
                       cmap='PuOr_r')
    axs[0, 2].contourf(channel_x, channel_z, y_target_normalized[:, 22, :, 2].T, vmin=-3, vmax=3,
                       cmap='PiYG_r')
    axs[1, 2].contourf(channel_x, channel_z, y_predict_normalized[:, 22, :, 2].T, vmin=-3, vmax=3,
                       cmap='PiYG_r')
    axs[0, 0].set_xlim([0, np.pi])
    axs[0, 0].set_ylim([0, np.pi / 2])
    axs[0, 1].set_xlim([0, np.pi])
    axs[0, 1].set_ylim([0, np.pi / 2])
    axs[0, 2].set_xlim([0, np.pi])
    axs[0, 2].set_ylim([0, np.pi / 2])
    axs[1, 0].set_xlim([0, np.pi])
    axs[1, 0].set_ylim([0, np.pi / 2])
    axs[1, 1].set_xlim([0, np.pi])
    axs[1, 1].set_ylim([0, np.pi / 2])
    axs[1, 2].set_xlim([0, np.pi])
    axs[1, 2].set_ylim([0, np.pi / 2])
    fig.show()
    # fig.savefig(self.generated_data_folder / contour_fig_name)
