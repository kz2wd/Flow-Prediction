
class PredictionSubSpace:
    def __init__(self, x_start=0, x_end=64, y_start=0, y_end=64, z_start=0, z_end=64):
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

    @property
    def x_size(self):
        return self.x[1] - self.x[0]

    @property
    def y_size(self):
        return self.y[1] - self.y[0]

    @property
    def z_size(self):
        return self.z[1] - self.z[0]


    def select(self, array):
        """
        Allow for overriding a value, for example if you want
        only first layer to make x, put y = 1.
        """
        return array[self.x[0]:self.x[1], self.y[0]:self.y[1], self.z[0]:self.z[1]]


    def sizes(self, x=None, y=None, z=None):
        """
       Allow for overriding a value, for example if you want
       only first layer to make x, put y = 1.
        """
        return (self.x_size if x is None else x,
                self.y_size if y is None else y,
                self.z_size if z is None else z)

