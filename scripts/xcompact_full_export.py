# Full pipelined script to import a dataset from xcompact to s3 and pgsql metadata
import argparse
from pathlib import Path

from dask.diagnostics import ProgressBar

import scripts.xcompact_utils as xc_utils
from scripts.database_add import add_dataset
from scripts.parser_utils import dir_path
from space_exploration.beans.channel_bean import Channel
from space_exploration.dataset import s3_access, db_access
from space_exploration.dataset.db_access import global_session

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-folder", required=True, type=dir_path, help="Folder containing the simulation")
    parser.add_argument("--dataset-name", required=True, type=str, help="the name of the dataset for it to be fetched from the code")
    parser.add_argument("--dataset-scaling", default=20, type=float, help="Arbitrary scaling value for the dataset, in the paper it was 100 / 3")
    parser.add_argument("--channel-name", required=True, type=str, help="Name of the channel that the dataset is in, will try to to find existing match, otherwise, will create a new one, check channel scale & input file")
    parser.add_argument("--channel-scale", default=200, type=float, help="arbitrary value for the y scale of the channel dimension to reach y+ = 200 at y= 1")
    parser.add_argument("--i3d-input-file", type=str, help="Use it if simulation input file is not in the simulation folders and name input.i3d")
    parser.add_argument("--y-lim", default=64, type=int,
                        help="limit of index to keep in y dimension")
    parser.add_argument("--auto", default=False, type=bool,
                        help="will not ask for user input if true")

    args = parser.parse_args()

    simulation_folder = Path(args.simulation_folder)

    # Resolve channel, either it already exist, just fetch it, otherwise create it, export it, use it
    channel_fetch = Channel.get_channel(args.channel_name)
    if channel_fetch is not None:
        channel = channel_fetch
        print(f"✅ Found existing channel [{args.channel_name}]")
    else:
        channel = xc_utils.add_channel_from_simulation(simulation_folder, args.channel_name, args.channel_scale, args.i3d_input_file)
        print(f"✅ Created new channel [{args.channel_name}]")

    x, y = xc_utils.get_snapshot_xy(simulation_folder, y_lim=args.y_lim)

    print(f"Prepared dataset of shape {y.shape}")

    chunk_size = xc_utils.get_chunk_size(y.shape[1:])
    y = y.rechunk(chunk_size)
    x = x.rechunk(chunk_size)

    print(f"Chunk size y:{y.chunksize}, x:{x.chunksize}")

    print("Building & Uploading dataset to S3 bucket")
    with ProgressBar():
        s3_access.store_xy(x, y, f"simulations/{args.dataset_name}.zarr")

    print("Exporting metadata to SQL database")
    dataset = add_dataset(args.dataset_name, args.dataset_scaling, channel)

    if not args.auto:
        input("Press [ENTER] to commit changes")
    try:
        global_session.commit()
    except Exception as e:
        print("Encountered exception, likely dataset metadata already exists, you may ignore this error")

    print("✅✅✅ Success ✅✅✅")
    print(f"Dataset [{args.dataset_name}] fully exported")

    if not args.auto:
        input("Press [ENTER] to proceed to benchmark")

    dataset.benchmark.compute()
