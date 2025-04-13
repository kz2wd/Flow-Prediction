import matplotlib.pyplot as plt

from models.GEN3D import GEN3D


class ModelBenchmarker:
    def __init__(self, models: list[GEN3D]):
        self.models: list[GEN3D] = models


    def compute_losses(self, distances: list[int] = None):
        if distances is None:
            distances = [15, 30, 50, 100, 200]  # default y_plus used in the paper

            for model in self.models:
                plt.plot(distances, [model.get_original_losses(y_plus)[0] / model.data_scaling for y_plus in distances], label=model.in_legend_name)
            plt.ylabel('U errors')
            plt.xlabel('Y+ distance')
            # plt.ylim([0, 10])
            plt.legend()
            plt.show()