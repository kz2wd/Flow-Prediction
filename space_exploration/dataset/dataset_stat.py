from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DatasetStats:
    u_means: np.array
    v_means: np.array
    w_means: np.array
    u_std: np.array
    v_std: np.array
    w_std: np.array