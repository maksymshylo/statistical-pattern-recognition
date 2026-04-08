import argparse
import numpy as np
from numba import njit, prange
from time import perf_counter

import matplotlib.pyplot as plt
from PIL import Image


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
                    + g[:, k]
                )
                P[i, j, 2, k] = max(
                    P[i - 1, j, 2, :]
                    + 0.5 * Q[i - 1, j, :]
                    + fi[i - 1, j, :]
                    + g[:, k]
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
    for i in range(height - 2, -1, -1):
        for j in range(width - 2, -1, -1):
            # for each label in a pixel
            for k in range(n_labels):
                # P[i, j, 1, k] - Right direction
                # P[i, j, 3, k] - Down direction
                # Calculate the best path weight
                P[i, j, 3, k] = max(
                    P[i + 1, j, 3, :]
                    + 0.5 * Q[i + 1, j, :]
                    + fi[i + 1, j, :]
                    + g[k, :]
                )
                P[i, j, 1, k] = max(
                    P[i, j + 1, 1, :]
                    + 0.5 * Q[i, j + 1, :]
                    - fi[i, j + 1, :]
                    + g[k, :]
                )
                # update potentials
                fi[i, j, k] = (
                    P[i, j, 0, k] + P[i, j, 1, k] - P[i, j, 2, k] - P[i, j, 3, k]
                ) * 0.5
    return P, fi


def inpaint_image(
    image: np.ndarray, labels: np.ndarray, alpha: float, epsilon: int, n_iter: int
) -> np.ndarray:
    """
    Inpaint the input image using TRW-S algorithm.

    Args:
        image: Input image. Shape: (height, width, channels).
        labels: Array of colors (mapping label->color).
        alpha: Smoothing coefficient.
        epsilon: Special parameter, which is responsible for lack of color information.
        n_iter: Number of iterations.

    Returns:
        Inpainted image
    """

    height, width, n_channels = image.shape
    n_labels = len(labels)
    inpainted_image = np.zeros_like(image)

    # Pre-compute binary penalties once (same for all channels)
    g = -alpha * np.abs(np.subtract(labels, labels.reshape(-1, 1))).astype(np.float64)

    # Process channels in parallel
    for channel in prange(n_channels):
        img_channel = image[:, :, channel]

        # Calculate unary penalties for this channel
        Q = -np.abs((img_channel[:, :, np.newaxis] - labels)) * (
            img_channel[:, :, np.newaxis] != epsilon
        )
        Q = Q.astype(np.float64)

        # Initialize P for this channel
        P = np.zeros((height, width, 4, n_labels), dtype=np.float64)

        # Run TRW-S for this channel
        fi = np.zeros((height, width, n_labels), dtype=np.float64)
        P, _ = backward_pass(height, width, n_labels, Q, g, P, fi.copy())

        for iteration in range(n_iter):
            P, fi = forward_pass(height, width, n_labels, Q, g, P, fi)
            P, fi = backward_pass(height, width, n_labels, Q, g, P, fi)

        # Restore labelling
        energy = P[:, :, 0, :] + P[:, :, 1, :] - fi + Q * 0.5
        labelling = np.argmax(energy, axis=2)
        # Map labels to values
        inpainted_image[:, :, channel] = labels[labelling]

    return inpainted_image


def main():
    parser = argparse.ArgumentParser(
        description="Image inpainter using TRW-S algorithm."
    )
    parser.add_argument(
        "--img_path", type=str, required=True, help="Path to the image."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Smoothing coefficient for binary penalties.",
    )
    parser.add_argument(
        "--epsilon",
        type=int,
        required=True,
        help="Special parameter, which is responsible for lack of color information.",
    )
    parser.add_argument("--n_labels", type=int, required=True, help="Number of labels.")
    parser.add_argument(
        "--n_iter", type=int, required=True, help="Number of iterations."
    )

    args = parser.parse_args()

    image = np.array(Image.open(args.img_path), dtype=np.uint8)

    t1 = perf_counter()
    labels = np.arange(0, 256, int(256 / args.n_labels))

    if type(args.n_iter) is not int or args.n_iter <= 0:
        raise Exception("Wrong n_iter parameter")

    if image.shape[0] <= 1 or image.shape[1] <= 1:
        raise Exception("image is empty")

    inpainted_image = inpaint_image(
        image, labels, args.alpha, args.epsilon, args.n_iter
    )

    t2 = perf_counter()
    print("Total time", t2 - t1)

    plt.imsave("inpainted.png", inpainted_image.astype(np.uint8))


if __name__ == "__main__":
    main()
