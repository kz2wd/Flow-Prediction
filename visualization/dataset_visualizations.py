import plotly.express as px

DATASET_VISUALIZATIONS = {}


def visualization(name):
    def decorator(func):
        DATASET_VISUALIZATIONS[name] = func
        return func
    return decorator


@visualization("Compare Dataset Sizes")
def compare_dataset_sizes(datasets):
    names = [ds.name for ds in datasets]
    sizes = [ds.load_s3().shape[0] for ds in datasets]
    fig = px.bar(x=names, y=sizes)
    fig.update_layout(title="Dataset Sizes")
    return fig

@visualization("U Velocities Along Y")
def u_velo_along_y(datasets):
    velocity_mean = self.ds[:, 0].mean(axis=(0, 1, 3)).compute()

    names = [ds.name for ds in datasets]
    velocities = [ds.load_s3().shape[0] for ds in datasets]
    fig = px.bar(x=names, y=sizes)
    fig.update_layout(title="Dataset Sizes")
    return fig



