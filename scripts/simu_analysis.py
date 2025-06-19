from argparse import ArgumentParser

from space_exploration.beans.dataset_bean import Dataset

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset", required=True)
    args = arg_parser.parse_args()

    target_dataset = args.dataset

    dataset = Dataset.get_dataset_or_fail(target_dataset)

    print(dataset.analysis_df)