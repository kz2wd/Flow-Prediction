
import os
from pathlib import Path

import argparse

import h5py

import s3_access

import zarr
import dask.array as da
from dask.diagnostics import ProgressBar


def hdf5_path(string):
    if os.path.isfile(string) and string.endswith('.hdf5'):
        return string
    else:
        raise ValueError(f'{string} is not a hdf5 file')

def upload_hdf5_to_s3(hdf5: Path, remote_destination: str,):
    print("Loading hdf5 file...")
    with h5py.File(hdf5, "r") as f:
        print("Hdf5 file contain keys:")
        print(f.keys())
        key_choice = input("Type wanted key:")
        dset = f[key_choice]
        darr = da.from_array(dset, chunks="auto")

        print(f"Uploading to s3://{remote_destination}...")
        s3_store = s3_access.get_s3_map(remote_destination)
        with ProgressBar():
            darr.to_zarr(s3_store, overwrite=True)
        print(f"✅ Uploaded to s3://{remote_destination}")


if __name__ == "__main__":
    print("Ensure hdf5 file fit in memory.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5', type=hdf5_path, required=True, help='path to local hdf5 file')
    parser.add_argument('--s3', type=str, required=True, help='Target location on S3 server')
    args = parser.parse_args()
    hdf5_file = Path(args.hdf5)
    upload_hdf5_to_s3(hdf5_file, args.s3)
