import matplotlib.pyplot as plt
import numpy as np

from space_exploration.data_viz import Plotter
from space_exploration.models.A.A03 import A03
from space_exploration.models.C.C04 import C04
from space_exploration.models.C.C05 import C05
from space_exploration.models.C.C06 import C06
from space_exploration.models.C.CBase import CBase


def test():
    model = C04()
    model.train(1, 1, 4)
    # model.lazy_test(10)
    # model.benchmark()
    # Plotter.plot_mse(model)

def test_multiple():
    models = [C04(), C05(), C06()]
    for model in models:
        model.test(5)

if __name__ == '__main__':
    # compare_data()
    # test_multiple()
    test()