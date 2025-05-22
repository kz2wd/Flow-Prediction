from strenum import StrEnum

import h5py

from space_exploration.FolderManager import FolderManager


class PlotData(StrEnum):
    total_mse_y_wise = 'total_mse_y_wise'
    u_mse_y_wise = 'u_mse_y_wise'
    v_mse_y_wise = 'v_mse_y_wise'
    w_mse_y_wise = 'w_mse_y_wise'


def save_benchmarks(model, data_dict: dict[PlotData, any]):
    benchmark_dataset = FolderManager.benchmark_file(model)
    with h5py.File(benchmark_dataset, 'a') as f:
        for name, data in data_dict.items():
            dataset_name = str(name)
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(dataset_name, data=data, compression='gzip')


def get_benchmarks(model, names: list[PlotData]):
    with h5py.File(FolderManager.benchmark_file(model), 'r') as f:
        return [f[str(name)][...] for name in names]

