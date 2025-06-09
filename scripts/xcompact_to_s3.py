import argparse
from pathlib import Path

from dask.diagnostics import ProgressBar

from scripts.parser_utils import dir_path
from scripts.xcompact_utils import get_snapshot_xy
from space_exploration.dataset import s3_access

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=dir_path, required=True)
    parser.add_argument('--zarr_store', type=str, required=True, help='Local path or s3://bucket/key')
    parser.add_argument('--use_s3', action='store_true')
    args = parser.parse_args()

    main_folder = Path(args.path)

    with ProgressBar():
        x, y = get_snapshot_xy(main_folder)

    with ProgressBar():
        s3_access.store_xy(x, y, args.zarr_store)

    print(f"Saved dataset to s3://{args.zarr_store} ✅")
