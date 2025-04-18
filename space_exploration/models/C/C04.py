from space_exploration.models.C.CBase import CBase


class C04(CBase):
    def __init__(self):
        super().__init__(name="C04", checkpoint="ckpt-1", up_sampling_indices=[6, 12, 18, 24, 30])