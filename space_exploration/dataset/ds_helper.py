
def unnormalize(ds, u_means, v_means, w_means, u_stds, v_stds, w_stds):
    components = [0, 1, 2]  # channel indices: u, v, w
    stds = [u_stds, v_stds, w_stds]
    means = [u_means, v_means, w_means]
    for c, std, mean in zip(components, stds, means):
        std = std[None, None, :, None]  # Casting into correct shape
        mean = mean[None, None, :, None]
        ds[:, c, :, :, :] = ds[:, c, :, :, :] * std + mean

    return ds


def normalize(ds, u_means, v_means, w_means, u_stds, v_stds, w_stds):
    components = [0, 1, 2]  # channel indices: u, v, w
    stds = [u_stds, v_stds, w_stds]
    means = [u_means, v_means, w_means]
    for c, std, mean in zip(components, stds, means):
        std = std[None, None, :, None]  # Casting into correct shape
        mean = mean[None, None, :, None]
        ds[:, c, :, :, :] = (ds[:, c, :, :, :] - mean) / std

    return ds

