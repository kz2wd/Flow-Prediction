import dask.array as da

import s3_access

if __name__ == "__main__":
    fs = s3_access.fs
    target_bucket = 'simulations'
    target_dataset = 'paper-dataset.zarr'

    ds = da.from_zarr(f"s3://{target_bucket}/{target_dataset}")
    print(ds)