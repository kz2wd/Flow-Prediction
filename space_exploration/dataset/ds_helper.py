
def y_along_component_denormalize(ds, stats):
    components = [0, 1, 2]  # channel indices: u, v, w
    stds = [stats.u_stds, stats.v_stds, stats.w_stds]
    means = [stats.u_means, stats.v_means, stats.w_means]
    for c, std, mean in zip(components, stds, means):
        std = std[None, None, :, None]  # Casting into correct shape
        mean = mean[None, None, :, None]
        ds[:, c, :, :, :] = ds[:, c, :, :, :] * std + mean

    return ds


def y_along_component_normalize(ds, stats):
    components = [0, 1, 2]  # channel indices: u, v, w
    stds = [stats.u_stds, stats.v_stds, stats.w_stds]
    means = [stats.u_means, stats.v_means, stats.w_means]
    for c, std, mean in zip(components, stds, means):
        std = std[None, None, :, None]  # Casting into correct shape
        mean = mean[None, None, :, None]
        ds[:, c, :, :, :] = (ds[:, c, :, :, :] - mean) / std

    return ds



