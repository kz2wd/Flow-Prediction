from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt

from space_exploration.simulation_channel.SimulationChannel import SimulationChannel


class DatasetAnalyzer:
    def __init__(self, ds, dataset_scaling, channel: SimulationChannel):
        self.channel = channel
        self.ds = ds
        self.dataset_scaling = dataset_scaling
        self.max_y = self.ds.shape[3]  # B 3 X Y Z

    def plot_u_velo_along_y(self, fig_name):
        with ProgressBar():
            velocity_mean = self.ds[:, 0].mean(axis=(0, 1, 3)).compute()

        # Ignore first point as it is an outlier
        plt.semilogx( self.channel.y_dimension[1:self.max_y] * self.channel.y_scale_to_y_plus,
                      velocity_mean[1:] * self.dataset_scaling, label="mean velocity")
        plt.title("mean stream wise velocity along wall-normal")
        plt.legend()
        plt.grid(True)
        plt.savefig(fig_name)
        print(f"✅ Saved figure [{fig_name}]!")

