import argparse
from pathlib import Path

from scripts.database_add import add_channel
from scripts.parser_utils import dir_path
from scripts.xcompact_utils import add_channel_from_simulation
from space_exploration.dataset import db_access


if __name__ == '__main__':
    session = db_access.get_session()

    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-path", required=True, type=dir_path)
    parser.add_argument("--channel-name", required=True, type=str)
    parser.add_argument("--channel-scale", required=False, default=1, type=float)
    parser.add_argument("--input-file", type=str)

    args = parser.parse_args()

    simulation_folder = Path(args.simulation_path)
    add_channel_from_simulation(session, simulation_folder, args.channel_name, args.channel_scale, args.input_file)

    session.commit()

    print(f"Channel [{args.channel_name}] exported ✅")