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

    def plot_stds(self, fig_name):
        with ProgressBar():
            velocity_stds = self.ds.std(axis=(2, 4)).mean(axis=0).compute()

        # Ignore first point as it is an outlier
        plt.plot(self.channel.y_dimension[1:self.max_y] * self.channel.y_scale_to_y_plus,
                     velocity_stds[0, 1:] * self.dataset_scaling, label="u+")
        plt.plot(self.channel.y_dimension[1:self.max_y] * self.channel.y_scale_to_y_plus,
                 velocity_stds[1, 1:] * self.dataset_scaling, label="v+")
        plt.plot(self.channel.y_dimension[1:self.max_y] * self.channel.y_scale_to_y_plus,
                 velocity_stds[2, 1:] * self.dataset_scaling, label="w+")
        plt.title("Standard deviation of the velocity components along the wall-normal")
        plt.legend()
        plt.grid(True)
        plt.savefig(fig_name)
        print(f"✅ Saved figure [{fig_name}]!")

    def plot_velocity_fluctuation(self, fig_name):

        velocity_mean = self.ds.mean(axis=(0, 2, 4))
        fluctuation_mean = self.ds - velocity_mean[None, :, None, :, None]
        with ProgressBar():
            squared_velocity_mean = (fluctuation_mean ** 2).mean(axis=(0, 2, 4)).compute()

        plt.plot(self.channel.y_dimension[1:self.max_y] * self.channel.y_scale_to_y_plus,
                 squared_velocity_mean[0, 1:] * self.dataset_scaling, label="u2")
        plt.plot(self.channel.y_dimension[1:self.max_y] * self.channel.y_scale_to_y_plus,
                 squared_velocity_mean[1, 1:] * self.dataset_scaling, label="v2")
        plt.plot(self.channel.y_dimension[1:self.max_y] * self.channel.y_scale_to_y_plus,
                 squared_velocity_mean[2, 1:] * self.dataset_scaling, label="w2")
        plt.title("Mean-squared velocity fluctuations")
        plt.legend()
        plt.grid(True)
        plt.savefig(fig_name)
        print(f"✅ Saved figure [{fig_name}]!")

