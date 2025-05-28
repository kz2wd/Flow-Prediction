import dask.array as da
import pyarrow.parquet as pq
import s3fs

# Set it in ~/.aws/credentials:
"""
[default]
aws_access_key_id = your-access-key
aws_secret_access_key = your-secret-key
"""

fs = s3fs.S3FileSystem(profile='default', client_kwargs={
    'endpoint_url': 'http://localhost:9000'  # Minio url here
})


def get_s3_map(file_path):
    return s3fs.S3Map(root=file_path, s3=fs, check=False)


def get_ds(file_path):
    return da.from_zarr(get_s3_map(file_path))


def store_ds(ds, file_path):
    store = get_s3_map(file_path)
    ds.to_zarr(store, overwrite=True)


def store_df(df, file_path):
    with fs.open(file_path, 'wb') as f:
        df.to_parquet(f,
                      compression='snappy',
                      engine='pyarrow',
                      index=False)


def load_df(file_path):
    return pq.ParquetDataset(file_path, filesystem=fs).read_pandas().to_pandas()


def exist(file_path):
    return fs.exists(file_path)
