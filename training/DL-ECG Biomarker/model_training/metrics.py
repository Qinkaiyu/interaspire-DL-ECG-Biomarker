from __future__ import annotations

import numpy as np
from lifelines.utils import concordance_index


def harrell_c_index(
    time: np.ndarray,
    event: np.ndarray,
    log_risk: np.ndarray,
) -> float:
    """Harrell C-index where a larger model score denotes higher risk."""
    return float(concordance_index(time, -log_risk, event))
