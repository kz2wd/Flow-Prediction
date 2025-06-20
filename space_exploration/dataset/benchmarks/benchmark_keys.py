from strenum import StrEnum

class DatasetBenchmarkKeys(StrEnum):
    VELOCITY_MEAN_ALONG_Y = "velocity_mean_along_y"
    VELOCITY_STD_ALONG_Y = "velocity_std_along_y"
    FLUCTUATION_ALONG_Y = "fluctuation_along_y"
    SQUARED_VELOCITY_MEAN_ALONG_Y = "squared_velocity_mean_along_y"
    REYNOLDS_UV = "reynolds_uv"