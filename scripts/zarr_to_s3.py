from pathlib import Path

import argparse

from scripts.parser_utils import zarr_path
from space_exploration.dataset import s3_access

import dask.array as da
from dask.diagnostics import ProgressBar



def upload_zarr_to_s3(zarr: Path, remote_destination: str,):
    print("Loading zarr file...")
    dset = da.from_zarr(str(zarr))
    darr = da.asarray(dset, chunks="auto")

    print(f"Uploading to s3://{remote_destination}...")
    s3_store = s3_access.get_s3_map(remote_destination)
    with ProgressBar():
        darr.to_zarr(s3_store, overwrite=True)
    print(f"✅ Uploaded to s3://{remote_destination}")


if __name__ == "__main__":
    print("Ensure hdf5 file fit in memory.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr', type=zarr_path, required=True, help='path to local zarr file')
    parser.add_argument('--s3', type=str, required=True, help='Target location on S3 server')
    args = parser.parse_args()
    zarr_file = Path(args.zarr)
    upload_zarr_to_s3(zarr_file, args.s3)
