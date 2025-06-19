
import vtk

from space_exploration.FolderManager import FolderManager
from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset import db_access


def main():
    target_dataset = "re200-sr001etot"
    ds = Dataset.get_dataset_or_fail(target_dataset)
    ds.export_frame_vts(0)


if __name__ == "__main__":
    main()