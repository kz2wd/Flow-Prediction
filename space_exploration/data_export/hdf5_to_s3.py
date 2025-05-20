import os
from pathlib import Path

import argparse

from space_exploration.data_export import s3_access


def hdf5_path(string):
    if os.path.isfile(string) and string.endswith('.hdf5'):
        return string
    else:
        raise ValueError(f'{string} is not a hdf5 file')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5', type=hdf5_path, required=True, help='path to local hdf5 file')
    parser.add_argument('--s3', type=str, required=True, help='Target location on S3 server')
    args = parser.parse_args()
    hdf5_file = Path(args.hdf5)

    # Upload file
    with open(hdf5_file, 'rb') as local_hdf5:
        with s3_access.fs.open(args.s3, 'wb') as s3_remote:
            s3_remote.write(local_hdf5.read())

    print(f"✅ Uploaded {hdf5_file} to s3://{args.s3}")



