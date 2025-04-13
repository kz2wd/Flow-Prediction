from models.A03 import A03
from models.ModelBenchmarker import ModelBenchmarker


# This is a main
# For now, just used as the root folder of the project


def compare_data():
    models = [
        A03(),
        A03(in_legend_name="A03-zeros"),
        A03(in_legend_name="A03-noise"),
    ]
    for model in models:
        model.lazy_predict(50)
    models[1].replace_prediction_with_zeros()
    models[2].replace_prediction_with_noise()
    benchmark = ModelBenchmarker(models)
    benchmark.compute_losses()


def test():
    model = A03()
    model.lazy_predict(50)


if __name__ == '__main__':
    compare_data()
    # test()