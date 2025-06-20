import argparse
from pathlib import Path

from scripts.parser_utils import dir_path
from scripts.xcompact_utils import add_channel_from_simulation
from space_exploration.dataset.db_access import global_session

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-path", required=True, type=dir_path)
    parser.add_argument("--channel-name", required=True, type=str)
    parser.add_argument("--channel-scale", required=False, default=1, type=float)
    parser.add_argument("--input-file", type=str)

    args = parser.parse_args()

    simulation_folder = Path(args.simulation_path)
    add_channel_from_simulation(simulation_folder, args.channel_name, args.channel_scale, args.input_file)

    global_session.commit()

    print(f"Channel [{args.channel_name}] exported ✅")