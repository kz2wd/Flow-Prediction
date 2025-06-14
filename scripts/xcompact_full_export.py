# Full pipelined script to import a dataset from xcompact to s3 and pgsql metadata
import argparse
from pathlib import Path

from dask.diagnostics import ProgressBar

import scripts.xcompact_utils as xc_utils
from scripts.parser_utils import dir_path
from space_exploration.beans.channel_bean import Channel
from space_exploration.dataset import s3_access, db_access


if __name__ == "__main__":
    session = db_access.get_session()

    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-folder", required=True, type=dir_path, help="Folder containing the simulation")
    parser.add_argument("--dataset-name", required=True, type=str, help="the name of the dataset for it to be fetched from the code")
    parser.add_argument("--dataset-scaling", default=20, type=float, help="Arbitrary scaling value for the dataset, in the paper it was 100 / 3")
    parser.add_argument("--channel-name", required=True, type=str, help="Name of the channel that the dataset is in, will try to to find existing match, otherwise, will create a new one, check channel scale & input file")
    parser.add_argument("--channel-scale", default=200, type=float, help="arbitrary value for the y scale of the channel dimension to reach y+ = 200 at y= 1")
    parser.add_argument("--i3d-input-file", type=str, help="Use it if simulation input file is not in the simulation folders and name input.i3d")

    args = parser.parse_args()

    simulation_folder = Path(args.simulation_folder)

    # Resolve channel, either it already exist, just fetch it, otherwise create it, export it, use it
    channel_fetch = Channel.get_channel(session, args.channel_name)
    if channel_fetch is not None:
        channel = channel_fetch
        print(f"✅ Found existing channel [{args.channel_name}]")
    else:
        channel = xc_utils.add_channel_from_simulation(session, simulation_folder, args.channel_name, args.channel_scale, args.i3d_input_file)
        print(f"✅ Created new channel [{args.channel_name}]")

    x, y = xc_utils.get_snapshot_xy(simulation_folder)

    print(f"Prepared dataset of shape {y.shape}")
    done = False
    while not done:
        user_in = input("Input new Y shape:")
        try:
            y_lim = int(user_in)
            done = True
        except ValueError:
            print(f"{user_in} is an invalid value")

    print(f"Shrinking y to {y_lim}")

    y = y[..., :y_lim, :]
    print(f"New shape y: {y.shape}, x: {x.shape}")

    print("Building & Uploading dataset to S3 bucket")
    with ProgressBar():
        s3_access.store_xy(x, y, f"simulations/{args.dataset_name}.zarr")

    print("Exporting metadata to SQL database")
    xc_utils.build_export_metadata(session, y, args.dataset_name, args.dataset_scaling, channel)

    input("Press [ENTER] to commit changes")
    session.commit()

    print("✅✅✅ Success ✅✅✅")
    print(f"Dataset [{args.dataset_name}] fully exported")