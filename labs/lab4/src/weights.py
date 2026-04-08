import numpy as np
from numba import njit, prange

from .gmm import gmm


@njit(fastmath=True, parallel=True, cache=True)
def calculate_q(image, n_bg, n_fg, bg_params, fg_params, Q) -> np.ndarray:
    """
    Calculate unary penalties using GMM likelihood.

    Args:
        image: Input image, shape (height, width, 3)
        n_bg: Number of background GMM components
        n_fg: Number of foreground GMM components
        bg_params: (det_cov, weights, inv_cov, means) for background
        fg_params: (det_cov, weights, inv_cov, means) for foreground
        Q: Preallocated penalty array, shape (height, width, 2)

    Returns:
        Q: Updated unary penalties
    """
    height, width, _ = image.shape
    bg_det_cov, bg_weights, bg_inv_cov, bg_means = bg_params
    fg_det_cov, fg_weights, fg_inv_cov, fg_means = fg_params

    # Precompute normalization constant (3D Gaussian)
    c = (2 * np.pi) ** (-1.5)

    # Process pixels in parallel
    for i in prange(height):
        for j in range(width):
            pixel = image[i, j, :]

            # Background likelihood
            bg_likelihood = 0.0
            for n in range(n_bg):
                diff = pixel - bg_means[n]
                # Mahalanobis distance: diff^T * inv_cov * diff
                mahal_dist = 0.0
                for k in range(3):
                    for m in range(3):
                        mahal_dist += diff[k] * bg_inv_cov[n, k, m] * diff[m]

                bg_likelihood += (
                    bg_weights[n] * bg_det_cov[n] * np.exp(-0.5 * mahal_dist)
                )

            # Foreground likelihood
            fg_likelihood = 0.0
            for n in range(n_fg):
                diff = pixel - fg_means[n]
                mahal_dist = 0.0
                for k in range(3):
                    for m in range(3):
                        mahal_dist += diff[k] * fg_inv_cov[n, k, m] * diff[m]

                fg_likelihood += (
                    fg_weights[n] * fg_det_cov[n] * np.exp(-0.5 * mahal_dist)
                )

            # Convert to log penalties
            Q[i, j, 0] = np.log(c * bg_likelihood + 1e-10)
            Q[i, j, 1] = np.log(c * fg_likelihood + 1e-10)

    return Q


@njit(fastmath=True, cache=True)
def calculate_beta(image: np.ndarray) -> float:
    """
    Calculate beta parameter from average gradient magnitude.

    Args:
        image: Input image, shape (height, width, 3)

    Returns:
        beta: Average squared color difference between neighbors
    """
    height, width, _ = image.shape
    beta = 0.0

    # Number of edges in a 4-connected grid
    tau = 2 * height * width - height - width

    # Accumulate squared differences
    for i in range(height):
        for j in range(width):
            # Right neighbor
            if j < width - 1:
                diff = image[i, j, :] - image[i, j + 1, :]
                beta += diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]

            # Down neighbor
            if i < height - 1:
                diff = image[i, j, :] - image[i + 1, j, :]
                beta += diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]

    return beta / tau


@njit(fastmath=True, parallel=True, cache=True)
def calculate_g(image, gamma, beta, g) -> np.ndarray:
    """
    Calculate binary penalties.


    Args:
        image: Input image, shape (height, width, 3)
        gamma: Smoothness weight
        beta: Normalization factor
        g: Preallocated penalty array, shape (height, width, 4, 2, 2)

    Returns:
        g: Updated pairwise penalties
    """
    height, width, _ = image.shape
    inv_2beta = 1.0 / (2.0 * beta)

    # Process pixels in parallel
    for i in prange(height):
        for j in range(width):
            pixel = image[i, j, :]

            # Left neighbor (if exists)
            if j > 0:
                diff = pixel - image[i, j - 1, :]
                dist_sq = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]
                penalty = -gamma * np.exp(-dist_sq * inv_2beta)
                g[i, j, 0, 0, 1] = penalty
                g[i, j, 0, 1, 0] = penalty

            # Right neighbor (if exists)
            if j < width - 1:
                diff = pixel - image[i, j + 1, :]
                dist_sq = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]
                penalty = -gamma * np.exp(-dist_sq * inv_2beta)
                g[i, j, 1, 0, 1] = penalty
                g[i, j, 1, 1, 0] = penalty

            # Up neighbor (if exists)
            if i > 0:
                diff = pixel - image[i - 1, j, :]
                dist_sq = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]
                penalty = -gamma * np.exp(-dist_sq * inv_2beta)
                g[i, j, 2, 0, 1] = penalty
                g[i, j, 2, 1, 0] = penalty

            # Down neighbor (if exists)
            if i < height - 1:
                diff = pixel - image[i + 1, j, :]
                dist_sq = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]
                penalty = -gamma * np.exp(-dist_sq * inv_2beta)
                g[i, j, 3, 0, 1] = penalty
                g[i, j, 3, 1, 0] = penalty

    return g


def calculate_penalties(
    image, gamma, beta, K, fg, bg, n_fg, n_bg, em_n_iter
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate both unary and binary penalties.

    Args:
        image: Input image, shape (height, width, 3)
        gamma: Smoothness weight
        beta: Gradient normalization factor
        K: Label set [0, 1]
        fg: Foreground samples, shape (n_fg_samples, 3)
        bg: Background samples, shape (n_bg_samples, 3)
        n_fg: Number of foreground GMM components
        n_bg: Number of background GMM components
        em_n_iter: EM iterations for GMM fitting

    Returns:
        Q: Unary penalties, shape (height, width, 2)
        g: Pairwise penalties, shape (height, width, 4, 2, 2)
    """
    height, width, _ = image.shape

    # Fit GMMs to samples
    fg_means, fg_sigma, _, fg_weights = gmm(fg, n_fg, n_iter=em_n_iter)
    bg_means, bg_sigma, _, bg_weights = gmm(bg, n_bg, n_iter=em_n_iter)

    # Batch determinant calculation
    fg_det_cov = np.array([np.linalg.det(sigma) ** (-0.5) for sigma in fg_sigma])
    bg_det_cov = np.array([np.linalg.det(sigma) ** (-0.5) for sigma in bg_sigma])

    # Batch inverse calculation
    fg_inv_cov = np.array([np.linalg.inv(sigma) for sigma in fg_sigma])
    bg_inv_cov = np.array([np.linalg.inv(sigma) for sigma in bg_sigma])

    # Convert means list to array
    fg_means = np.array(fg_means)
    bg_means = np.array(bg_means)

    # Package parameters
    bg_params = (bg_det_cov, bg_weights, bg_inv_cov, bg_means)
    fg_params = (fg_det_cov, fg_weights, fg_inv_cov, fg_means)

    # Preallocate arrays
    Q = np.empty((height, width, len(K)), dtype=np.float64)
    g = np.zeros((height, width, 4, len(K), len(K)), dtype=np.float64)

    # Calculate penalties
    Q = calculate_q(image, n_bg, n_fg, bg_params, fg_params, Q)
    g = calculate_g(image, gamma, beta, g)

    return Q, g
