from __future__ import annotations

import numpy as np

from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace


class ChannelData:
    def __init__(self, source_file):
        # Load mean velocity values in the streamwise and wall-normal directions for low- and high-resolution data
        with np.load(source_file) as f:
            self.U_mean = np.expand_dims(f['U_mean'], axis=-1)[:, :64, :, :]
            self.V_mean = np.expand_dims(f['V_mean'], axis=-1)[:, :64, :, :]
            self.W_mean = np.expand_dims(f['W_mean'], axis=-1)[:, :64, :, :]
            self.U_std = np.expand_dims(f['U_std'], axis=-1)[:, :64, :, :]
            self.V_std = np.expand_dims(f['V_std'], axis=-1)[:, :64, :, :]
            self.W_std = np.expand_dims(f['W_std'], axis=-1)[:, :64, :, :]
            self.PB_mean = np.expand_dims(f['PB_mean'], axis=-1)
            self.TBX_mean = np.expand_dims(f['TBX_mean'], axis=-1)
            self.TBZ_mean = np.expand_dims(f['TBZ_mean'], axis=-1)
            self.PB_std = np.expand_dims(f['PB_std'], axis=-1)
            self.TBX_std = np.expand_dims(f['TBX_std'], axis=-1)
            self.TBZ_std = np.expand_dims(f['TBZ_std'], axis=-1)


