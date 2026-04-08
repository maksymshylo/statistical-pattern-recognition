import numpy as np
from numba import njit, prange


@njit(fastmath=True, cache=True, parallel=True)
def forward_pass(
    height: int,
    width: int,
    n_labels: int,
    Q: np.ndarray,
    g: np.ndarray,
    P: np.ndarray,
    fi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward pass of TRW-S algorithm (Left and Up directions).

    Args:
        height: Height of input image
        width: Width of input image
        n_labels: Number of labels in labelset
        Q: Unary penalties
        g: Binary penalties
        P: Array of the best path weights for each direction (Left, Right, Up,Down)
        fi: Array of potentials

    Returns:
        Updated P and fi arrays.

    """
    # Go from the top-left to the bottom-right pixel
    for i in prange(1, height):
        for j in range(1, width):
            # for each label in pixel
            for k in range(n_labels):
                # P[i, j, 0, k] - Left direction
                # P[i, j, 2, k] - Up direction
                # calculate the best path weight according to formula
                P[i, j, 0, k] = max(
                    P[i, j - 1, 0, :]
                    + 0.5 * Q[i, j - 1, :]
                    - fi[i, j - 1, :]
                    + g[i, j - 1, 1, :, k]
                )
                P[i, j, 2, k] = max(
                    P[i - 1, j, 2, :]
                    + 0.5 * Q[i - 1, j, :]
                    + fi[i - 1, j, :]
                    + g[i - 1, j, 3, :, k]
                )
                # Update potentials
                fi[i, j, k] = (
                    P[i, j, 0, k] + P[i, j, 1, k] - P[i, j, 2, k] - P[i, j, 3, k]
                ) * 0.5
    return P, fi


@njit(fastmath=True, cache=True)
def backward_pass(
    height: int,
    width: int,
    n_labels: int,
    Q: np.ndarray,
    g: np.ndarray,
    P: np.ndarray,
    fi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backward pass of TRW-S algorithm (Right and Down directions).

    Args:
        height: Height of input image
        width: Width of input image
        n_labels: Number of labels in labelset
        Q: Unary penalties
        g: Binary penalties
        P: Array of the best path weights for each direction (Left, Right, Up, Down)
        fi: Array of potentials

    Returns:
        Updated P and fi arrays.

    """
    # Go from the bottom-right to the top-left pixel
    for i in np.arange(height - 2, -1, -1):
        for j in np.arange(width - 2, -1, -1):
            # for each label in a pixel
            for k in range(n_labels):
                # P[i, j, 1, k] - Right direction
                # P[i, j, 3, k] - Down direction
                # Calculate the best path weight
                P[i, j, 3, k] = max(
                    P[i + 1, j, 3, :]
                    + 0.5 * Q[i + 1, j, :]
                    + fi[i + 1, j, :]
                    + g[i + 1, j, 2, k, :]
                )
                P[i, j, 1, k] = max(
                    P[i, j + 1, 1, :]
                    + 0.5 * Q[i, j + 1, :]
                    - fi[i, j + 1, :]
                    + g[i, j + 1, 0, k, :]
                )
                # update potentials
                fi[i, j, k] = (
                    P[i, j, 0, k] + P[i, j, 1, k] - P[i, j, 2, k] - P[i, j, 3, k]
                ) * 0.5
    return P, fi


def trws(
    height: int,
    width: int,
    n_labels: int,
    K: np.ndarray,
    Q: np.ndarray,
    g: np.ndarray,
    P: np.ndarray,
    n_iter: int,
) -> np.ndarray:
    """
    Perform TRW-S algorithm.

    Args:
        height: Height of input image
        width: Width of input image
        n_labels: Number of labels in labelset
        Q: Unary penalties
        g: Binary penalties
        P: Array of the best path weights for each direction (Left, Right, Up, Down)
        n_iter: number of iteratations
    Returns
        Optimal labelling (with color mapping).

    """
    if n_iter <= 0:
        raise Exception("n_iter <=0")
    if len(K) != n_labels:
        raise Exception("n_labels do not match with real number of labels")

    # init an array of potentials
    fi = np.zeros((height, width, n_labels))
    # init Right and Down directions
    P, _ = backward_pass(height, width, n_labels, Q, g, P, fi.copy())
    for _ in range(n_iter):
        P, fi = forward_pass(height, width, n_labels, Q, g, P, fi)
        P, fi = backward_pass(height, width, n_labels, Q, g, P, fi)
    # restore labelling from optimal energy after n_iter of TRW-S
    labelling = np.argmax(P[:, :, 0, :] + P[:, :, 1, :] - fi + Q * 0.5, axis=2)
    # mapping from labels to colors
    output = K[labelling]
    return output


def optimal_labelling(Q, g, K, n_iter):
    """
    Input parameters for TRW-S algorithm.
    Run TRW-S algorithm.


    Args:
        Q: Unary penalties
        g: Binary penalties
        K: ndarray
            set of labels
        n_iter: int
            number of iteratations
    Returns
        labelling: ndarray
        array of optimal labelling (with color mapping)
        for one channel in input image

    """

    height, width, _ = Q.shape
    n_labels = len(K)
    P = np.zeros((height, width, 4, n_labels))
    labelling = trws(height, width, n_labels, K, Q, g, P, n_iter)
    return labelling
