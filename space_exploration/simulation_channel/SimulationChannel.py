import numpy as np



class SimulationChannel:
    def __init__(self, x_length, x_resolution, z_length, z_resolution, y_dimension, y_scale_to_y_plus,
                 name="unnamed"):
        self.x_dimension = np.arange(x_resolution) * x_length / x_resolution
        self.y_dimension = y_dimension
        self.z_dimension = np.arange(z_resolution) * z_length / z_resolution
        self.y_scale_to_y_plus = y_scale_to_y_plus
        self.name = name