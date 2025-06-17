import argparse

from space_exploration.dataset import db_access

if __name__ == "__main__":
    session = db_access.get_session()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--x-transform", required=True, type=str)
    parser.add_argument("--y-transform", required=True, type=str)
    parser.add_argument("--size", required=False, type=int, default=-1)

    args = parser.parse_args()

    