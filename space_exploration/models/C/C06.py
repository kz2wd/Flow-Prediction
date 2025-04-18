from space_exploration.models.C.CBase import CBase


class C06(CBase):
    def __init__(self):
        super().__init__(name="C06", checkpoint_number="ckpt-20", up_sampling_indices=[10, 20, 30, 40, 50])
