import dask.array as da


def y_along_component_denormalize(ds, means, stds):
    components = [0, 1, 2]  # u, v, w

    y_lim = ds.shape[3]
    denorm_components = []
    for c, std, mean in zip(components, stds, means):
        std = std[None, None, :y_lim, None]
        mean = mean[None, None, :y_lim, None]
        component = ds[:, c, :, :, :] * std + mean
        denorm_components.append(component)

    # Stack back into shape (N, 3, x, y, z)
    return da.stack(denorm_components, axis=1)


def y_along_component_normalize(ds, means, stds):
    components = [0, 1, 2]  # channel indices: u, v, w

    y_lim = ds.shape[3]
    norm_components = []
    for c, std, mean in zip(components, stds, means):
        std = da.where(std == 0, 1e-8, std).compute()
        std = std[None, None, :y_lim, None]  # Casting into correct shape
        mean = mean[None, None, :y_lim, None]
        component = (ds[:, c, :, :, :] - mean) / std
        norm_components.append(component)

    # Stack back into shape (N, 3, x, y, z)
    return da.stack(norm_components, axis=1)



