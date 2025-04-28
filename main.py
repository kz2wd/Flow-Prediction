
from space_exploration.data_viz import Plotter
from space_exploration.data_viz.PlotData import PlotData
from space_exploration.models.A.A03 import A03
from space_exploration.models.C.C04 import C04
from space_exploration.models.C.C05 import C05
from space_exploration.models.C.C06 import C06


def test():

    model = A03()
    # model.lazy_test(10)
    # model.benchmark()
    Plotter.plot_mse(model, "mse")


if __name__ == '__main__':
    # test_multiple()
    test()