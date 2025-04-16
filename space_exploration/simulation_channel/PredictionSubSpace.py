

class PredictionSubSpace:
    def __init__(self,  x_start, x_end, y_start, y_end, z_start, z_end, ):
        self.x = (x_start, x_end)
        self.y = (y_start, y_end)
        self.z = (z_start, z_end)

    @property
    def x_slice(self):
        return slice(*self.x)

    @property
    def y_slice(self):
        return slice(*self.y)

    @property
    def z_slice(self):
        return slice(*self.z)
