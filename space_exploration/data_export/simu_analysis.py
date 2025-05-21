import dask.array as da
import zarr
from matplotlib import pyplot as plt

import s3_access



def plot_y_velo(ds, plot_name):

    velocity_mean = ds.norm(axis=4).mean(axis=(0, 1, 3)).compute()

    plt.semilogx(velocity_mean[..., 0])
    plt.title("velocity mean along wall-normal")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_name)




if __name__ == "__main__":
    fs = s3_access.fs
    target_bucket = 'simulations'
    target_dataset = 'paper-dataset.zarr'

    file_path = f"s3://{target_bucket}/{target_dataset}"
    store = s3_access.get_s3_map(file_path)
    z = zarr.open_array(store, mode='r')
    ds = da.from_zarr(z)
    print(ds)

    plot_y_velo(ds, "u_velo.png")




