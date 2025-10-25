from __future__ import annotations

import numpy as np
from typing import Tuple

from load_mnist_data import load_mnist_data


def preprocess_mnist(
    images: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    *,
    flatten: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened/normalized MNIST data.

    Parameters
    ----------
    images, labels
        Optional arrays. If omitted, the data is loaded via ``load_mnist_data``.
    flatten
        Whether to flatten the images into shape ``(n_samples, n_features)``.
    normalize
        Whether to min-max normalize each sample after flattening.
    """

    if images is None or labels is None:
        images, labels = load_mnist_data()

    if flatten:
        flat = images.reshape(images.shape[0], -1)
    else:
        flat = images

    if normalize:
        X_min = flat.min(axis=1, keepdims=True)
        X_max = flat.max(axis=1, keepdims=True)
        denom = np.where((X_max - X_min) == 0, 1.0, X_max - X_min)
        normalized = (flat - X_min) / denom
    else:
        normalized = flat

    return flat, normalized, labels