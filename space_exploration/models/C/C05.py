from space_exploration.models.C.CBase import CBase


class C05(CBase):
    def __init__(self):
        super().__init__(name="C05", checkpoint_number="ckpt-20", up_sampling_indices=[8, 16, 24, 32, 40])
