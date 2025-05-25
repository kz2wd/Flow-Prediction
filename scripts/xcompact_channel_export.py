import argparse
from pathlib import Path

from scripts.database_add import add_channel
from scripts.parser_utils import dir_path
from space_exploration.dataset import db_access


def read_ypi(folder):
    with open(folder / "ypi.dat", 'r') as f:
        return [float(line.strip()) for line in f.readlines()]

def get_channel_data(filepath):
    channel_data = {}
    with open(filepath, 'r') as file:
        for line in file:
            # Remove everything after '!' (comments)
            line = line.split('!')[0].strip()
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Try converting value to int, float, or keep as string
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                channel_data[key] = value
    return channel_data


if __name__ == '__main__':
    session = db_access.get_session()

    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-path", required=True, type=dir_path)
    parser.add_argument("--channel-name", required=True, type=str)
    parser.add_argument("--channel-scale", required=True, type=float)
    parser.add_argument("--input-file", type=str)

    args = parser.parse_args()

    simulation_folder = Path(args.simulation_path)

    if args.input_file:
        input_file = Path(args.input_file)
    else:
        input_file = simulation_folder / "input.i3d"

    channel_data = get_channel_data(input_file)
    y_dim = read_ypi(simulation_folder)

    add_channel(session,
                args.channel_name,
                channel_data['nx'], channel_data['xlx'],
                y_dim,
                channel_data['nz'], channel_data['zlz'],
                args.channel_scale)

    session.commit()

    print(f"Added channel [{args.channel_name}].")