import h5py
import matplotlib.pyplot as plt

from FolderManager import FolderManager


def plot_mse(source_model, fig_name):
    with h5py.File(FolderManager.benchmark_file(source_model), 'r') as f:
        u_mse = f['u_mse_y_wise'][...]
        v_mse = f['v_mse_y_wise'][...]
        w_mse = f['w_mse_y_wise'][...]


    plt.plot(u_mse, label='u_mse')
    plt.plot(v_mse, label='v_mse')
    plt.plot(w_mse, label='w_mse')
    plt.legend()
    plt.savefig(FolderManager.generated_data(source_model) / fig_name)
