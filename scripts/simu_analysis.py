import dask.array as da
import zarr
from matplotlib import pyplot as plt
from space_exploration.dataset import s3_access



def plot_y_velo(ds, plot_name):


    velocity_mean = ds[:, 0].mean(axis=(0, 1, 3)).compute()


    plt.semilogx(velocity_mean, label="mean velocity")
    plt.title("mean stream wise velocity along wall-normal")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_name)


if __name__ == "__main__":
    fs = s3_access.fs
    target_bucket = 'simulations'
    target_dataset = 'paper-dataset.zarr'

    file_path = f"s3://{target_bucket}/{target_dataset}"
    store = s3_access.get_s3_map(file_path)
    ds = da.from_zarr(store)
    print(ds)

    plot_y_velo(ds, "paper_y_velo.png")




