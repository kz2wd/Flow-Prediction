
from space_exploration.data_viz import Plotter
from space_exploration.models.A.A03 import A03
from space_exploration.models.C.C04 import C04
from space_exploration.models.C.C05 import C05
from space_exploration.models.C.C06 import C06


def test():
    model = C04()
    # model.train(1, 1, 4)
    # model.generate_pipeline_training()
    # model.generate_datasets("test", 1)
    model.test(20)
    # model.benchmark()
    # Plotter.plot_mse(model, "mse")

def test_multiple():
    models = [C04(), C05(), C06()]
    for model in models:
        model.test(5)

if __name__ == '__main__':
    # test_multiple()
    test()