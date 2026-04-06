import argparse
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from colordict import ColorDict
from numba import njit, prange
from PIL import Image


def calculate_unary_penalties(
    image: np.ndarray, colors: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Calculate unary penalties for image."""
    H, W, C = image.shape
    K = len(labels)
    # Flatten image to (N, 3) where N = H*W
    X = image.reshape(-1, C)
    C_subset = colors[labels]
    # Squared norm of image pixels: (N, 1)
    x_sq = np.sum(X**2, axis=1, keepdims=True)
    # Squared norm of colors: (K,)
    c_sq = np.sum(C_subset**2, axis=1)
    # Matrix multiplication for the dot product: (N, K)
    dot = X @ C_subset.T
    # Squared distance: (N, K)
    dist_sq = x_sq + c_sq - 2 * dot
    # Find best label per pixel
    mins = np.argmin(dist_sq, axis=1)
    # Create a boolean mask using one-hot indexing
    return np.eye(K, dtype=bool)[mins].reshape(H, W, K)


@njit(cache=True, fastmath=True)
def get_neighbours(
    height: int, width: int, i: int, j: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate neighbors in a 4-neighbors system (and inverse indices).

             (i-1, j)
                |
    (i,j-1) - (i, j) - (i, j+1)
                |
            (i+1, j)

    Args:
        height: Height of the image
        width: Width of the image
        i: Number of row
        j: Number of column

    Returns:
        Tuple of neighbors
            - coordinates
            - indices
            - inverse indices

    """
    if width <= 0 or height <= 0:
        raise Exception("height or width is less than zero")

    # If the center pixel is out of bounds, return empty arrays
    if i < 0 or i >= height or j < 0 or j >= width:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    # Pre-allocate for the maximum possible neighbors (4)
    nbs = np.empty((4, 2), dtype=np.int64)
    inv_nbs_indices = np.empty(4, dtype=np.int64)
    nbs_indices = np.empty(4, dtype=np.int64)

    count = 0

    # Left (Index: 0, Inverse: 1)
    if j > 0:
        nbs[count, 0], nbs[count, 1] = i, j - 1
        nbs_indices[count], inv_nbs_indices[count] = 0, 1
        count += 1

    # Right (Index: 1, Inverse: 0)
    if j < width - 1:
        nbs[count, 0], nbs[count, 1] = i, j + 1
        nbs_indices[count], inv_nbs_indices[count] = 1, 0
        count += 1

    # Upper (Index: 2, Inverse: 3)
    if i > 0:
        nbs[count, 0], nbs[count, 1] = i - 1, j
        nbs_indices[count], inv_nbs_indices[count] = 2, 3
        count += 1

    # Down (Index: 3, Inverse: 2)
    if i < height - 1:
        nbs[count, 0], nbs[count, 1] = i + 1, j
        nbs_indices[count], inv_nbs_indices[count] = 3, 2
        count += 1

    # Return slices of the arrays up to the number of valid neighbors found
    return nbs[:count], inv_nbs_indices[:count], nbs_indices[:count]


@njit(fastmath=True, cache=True, parallel=True)
def diffusion_iteration(
    height: int,
    width: int,
    labels: np.ndarray,
    potentials: np.ndarray,
    unary_pnts: np.ndarray,
    binary_pnts: np.ndarray,
) -> np.ndarray:
    num_labels = len(labels)

    # Pre-calculate neighbor offsets: (di, dj, idx, inv_idx)
    # 0: Left, 1: Right, 2: Upper, 3: Down
    offsets = np.array(
        [
            (0, -1, 0, 1),  # Left
            (0, 1, 1, 0),  # Right
            (-1, 0, 2, 3),  # Upper
            (1, 0, 3, 2),  # Down
        ]
    )

    for i in prange(height):
        # We create a small local buffer for labels to help the compiler optimize
        # This is reused across the width to avoid repeated allocations
        m_vals = np.zeros(4)

        for j in range(width):
            # 1. Identify valid neighbors once for this pixel
            valid_neighbors = []
            for o in range(4):
                ni, nj = i + offsets[o, 0], j + offsets[o, 1]
                if 0 <= ni < height and 0 <= nj < width:
                    # Store (ni, nj, inv_idx, nbs_idx)
                    valid_neighbors.append((ni, nj, offsets[o, 3], offsets[o, 2]))

            num_nbs = len(valid_neighbors)
            if num_nbs == 0:
                continue

            # 2. Iterate through labels
            for l_idx in range(num_labels):
                l = labels[l_idx]
                sum_m = 0.0

                # 3. Calculate Max-Sum (l_asterisk logic) for each neighbor
                for n in range(num_nbs):
                    ni, nj, inv_idx, _ = valid_neighbors[n]

                    # Inner-most optimization: Find max_k (binary[l, k] - potential[ni, nj, inv, k])
                    max_val = -1e18  # Representation of -infinity
                    for k_idx in range(num_labels):
                        val = binary_pnts[l, k_idx] - potentials[ni, nj, inv_idx, k_idx]
                        if val > max_val:
                            max_val = val

                    m_vals[n] = max_val
                    sum_m += max_val

                # 4. Calculate C_t_sum and update in-place
                c_t_sum = (sum_m + unary_pnts[i, j, l]) / num_nbs

                for n in range(num_nbs):
                    _, _, _, nbs_idx = valid_neighbors[n]
                    potentials[i, j, nbs_idx, l] = m_vals[n] - c_t_sum

    return potentials


def get_labelling(
    height: int,
    width: int,
    binary_pnts: np.ndarray,
    labels: np.ndarray,
    potentials: np.ndarray,
) -> np.ndarray:
    """Restore labeling from optimal energy after n_iter of diffusion.

    Args:
        height: Height of the image
        width: Width of the image
        binary_pnts: Binary penalties
        labels: Label set
        potentials: Potentials
    """

    labelling = np.empty((height, width, len(labels[0])), dtype=int)
    for i in range(height):
        for j in range(width):
            nbs, inv_nbs_indices, nbs_indices = get_neighbours(height, width, i, j)
            # take any neighbor
            n_i, n_j = nbs[0]
            # calculating reparametrized binary penalties
            g_reparametrized = (
                binary_pnts
                - potentials[i, j, nbs_indices[0], :]
                - potentials[n_i, n_j, inv_nbs_indices[0], :]
            )
            # binary penalties - is supermodular so take the highest possible maximum edge between nodes t, t'
            labelling[i, j, :] = labels[np.argmax(np.max(g_reparametrized, axis=0))]
    return labelling


def main():
    parser = argparse.ArgumentParser(
        description="Image segmentation on a noised image using diffusion."
    )
    parser.add_argument(
        "--img_path", type=str, required=True, help="Path to the image to denoise"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Alpha parameter for binary penalties",
    )
    parser.add_argument("--n_iter", type=int, help="Number of iterations")
    parser.add_argument(
        "--colors",
        type=str,
        required=True,
        help="List of colors  on the image to segment.",
    )

    args = parser.parse_args()

    image = np.array(
        Image.open(
            args.img_path,
        ),
        dtype=int,
    )

    n_neighbors = 4
    height, width, _ = image.shape
    color_dict = ColorDict()
    colors = np.array([color_dict[key] for key in args.colors.split(" ")], dtype=int)
    n_labels = len(colors)
    # defining label set
    labels = np.arange(n_labels)

    if colors.shape[0] < 2:
        raise Exception("less than two colors")
    if (labels != np.arange(colors.shape[0])).any():
        raise Exception("incorrect set of labels")
    if args.n_iter <= 0:
        raise Exception("number of iterations is less or equal to zero")

    t1 = perf_counter()
    # Get unary penalties
    unary_pnts = calculate_unary_penalties(image, colors, labels)

    # Get binary penalties
    binary_pnts = args.alpha * np.identity(n_labels)

    # Initialize potentials as zeros
    potentials = np.zeros((height, width, n_neighbors, n_labels))

    # Run diffusion iterations
    for i in range(args.n_iter):
        potentials = diffusion_iteration(
            height, width, labels, potentials, unary_pnts, binary_pnts
        )
    # Restore labelling
    labelling_img = get_labelling(height, width, binary_pnts, colors, potentials)

    t2 = perf_counter()
    plt.imsave("denoised.png", labelling_img.astype(np.uint8))
    print("Total time", t2 - t1)


if __name__ == "__main__":
    main()
