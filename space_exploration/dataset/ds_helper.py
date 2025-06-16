import dask.array as da


def y_along_component_denormalize(ds, stats):
    components = [0, 1, 2]  # u, v, w
    stds = [stats.u_stds, stats.v_stds, stats.w_stds]
    means = [stats.u_means, stats.v_means, stats.w_means]

    denorm_components = []
    for c, std, mean in zip(components, stds, means):
        std = std[None, None, :, None]
        mean = mean[None, None, :, None]
        component = ds[:, c, :, :, :] * std + mean
        denorm_components.append(component)

    # Stack back into shape (N, 3, x, y, z)
    return da.stack(denorm_components, axis=1)


def y_along_component_normalize(ds, stats):
    components = [0, 1, 2]  # channel indices: u, v, w
    stds = [stats.u_stds, stats.v_stds, stats.w_stds]
    means = [stats.u_means, stats.v_means, stats.w_means]

    print("stds:",)


    norm_components = []
    for c, std, mean in zip(components, stds, means):
        std = std[None, None, :, None]  # Casting into correct shape
        mean = mean[None, None, :, None]
        component = (ds[:, c, :, :, :] - mean) / std
        norm_components.append(component)

    # Stack back into shape (N, 3, x, y, z)
    return da.stack(norm_components, axis=1)



