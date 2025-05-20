import argparse
from pathlib import Path
import h5py
import zarr
import dask.array as da
from dask.diagnostics import ProgressBar
from mpmath.functions.hypergeometric import hyp3f2


def hdf5_path(string):
    path = Path(string)
    if path.is_file() and path.suffix == '.hdf5':
        return path
    raise argparse.ArgumentTypeError(f"{string} is not a valid .hdf5 file")


def convert_dataset(h5_dataset, zar_file, dataset_name):
    chunks = (10, *h5_dataset.shape[1:])
    dset = da.from_array(h5_dataset, chunks=chunks)
    print(f"Converting dataset '{dataset_name}' with shape {h5_dataset.shape} and chunks {chunks}")
    dset.to_zarr(url=str(zar_file))

def convert_hdf5_to_zarr(hdf5_path: Path, zarr_path: Path):
    with h5py.File(hdf5_path, 'r') as h5file:
        print(h5file.keys())
        chosen_key = input("Type key:")
        item = h5file[chosen_key]
        with ProgressBar():
            convert_dataset(item, zarr_path, chosen_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5', type=hdf5_path, required=True, help='Path to input HDF5 file')
    parser.add_argument('--zarr', type=Path, required=True, help='Output path for Zarr store')
    args = parser.parse_args()

    convert_hdf5_to_zarr(args.hdf5, args.zarr)
    print(f"✅ Zarr conversion complete: {args.zarr}")
