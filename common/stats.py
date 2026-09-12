"""Generic numeric helpers shared across stages: outlier-robust reductions and
smoothing kernels."""
import numpy as np


def mad_filter_median(arr, min_for_filter=5):
    """MAD-based outlier removal + median over arr (N, D).
    Returns the inlier median (D,), the all-rows median if N < min_for_filter, or None if N == 0."""
    N = arr.shape[0]
    if N == 0:
        return None
    med = np.median(arr, axis=0)
    if N < min_for_filter:
        return med
    mad = np.median(np.abs(arr - med), axis=0)
    sigma_hat = np.maximum(1.4826 * mad, 1e-8)
    thr = 4.0 if N <= 6 else (3.5 if N <= 9 else 3.0)
    keep = (np.abs(arr - med) / sigma_hat <= thr).all(axis=1)
    if not keep.any():
        return med
    return np.median(arr[keep], axis=0)


def gaussian_kernel(size, sigma):
    """Centered 1-D Gaussian kernel of length `size`, normalized to sum 1."""
    if size <= 0:
        raise ValueError("Kernel size must be positive")
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()
