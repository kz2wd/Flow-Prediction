from matplotlib import pyplot as plt

from space_exploration.simulation_channel.SimulationChannel import SimulationChannel


class DatasetAnalyzer:
    def __init__(self, ds, channel: SimulationChannel):
        self.channel = channel
        self.ds = ds

    def plot_u_velo_along_y(self, fig_name):
        velocity_mean = self.ds[:, 0].mean(axis=(0, 1, 3)).compute()

        plt.semilogx(velocity_mean, label="mean velocity")
        plt.title("mean stream wise velocity along wall-normal")
        plt.legend()
        plt.grid(True)
        plt.savefig(fig_name)

