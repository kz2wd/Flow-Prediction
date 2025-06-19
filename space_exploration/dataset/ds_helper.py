import dask.array as da


def y_along_component_denormalize(ds):
    components = [0, 1, 2]  # u, v, w
    sample_size = 100

    means = ds[sample_size:].mean(axis=(0, 2, 4)).compute()
    stds = ds[sample_size:].std(axis=(0, 2, 4)).compute()

    denorm_components = []
    for c, std, mean in zip(components, stds, means):
        std = std[None, None, :, None]
        mean = mean[None, None, :, None]
        component = ds[:, c, :, :, :] * std + mean
        denorm_components.append(component)

    # Stack back into shape (N, 3, x, y, z)
    return da.stack(denorm_components, axis=1)


def y_along_component_normalize(ds):
    components = [0, 1, 2]  # channel indices: u, v, w
    sample_size = 100

    means = ds[sample_size:].mean(axis=(0, 2, 4)).compute()
    stds = ds[sample_size:].std(axis=(0, 2, 4)).compute()

    norm_components = []
    for c, std, mean in zip(components, stds, means):
        std = da.where(std == 0, 1e-8, std).compute()
        std = std[None, None, :, None]  # Casting into correct shape
        mean = mean[None, None, :, None]
        component = (ds[:, c, :, :, :] - mean) / std
        norm_components.append(component)

    # Stack back into shape (N, 3, x, y, z)
    return da.stack(norm_components, axis=1)



