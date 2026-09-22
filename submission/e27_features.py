"""Feature function shipped inside the E27 length submission; the pickled model references it."""

import numpy as np


def log_length(problems):
    return np.log(np.asarray([len(p) for p in problems], dtype=float))[:, None]
