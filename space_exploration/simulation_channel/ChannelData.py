from __future__ import annotations

import numpy as np

from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace


class ChannelData:
    def __init__(self, u_means, v_means, w_means, u_stds, v_stds, w_stds):
        self.w_stds = w_stds
        self.v_stds = v_stds
        self.u_stds = u_stds
        self.w_means = w_means
        self.v_means = v_means
        self.u_means = u_means


