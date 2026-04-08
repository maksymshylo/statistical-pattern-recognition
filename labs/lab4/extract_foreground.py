"""
Foreground Extraction using Graph Cuts with EM and TRW-S

This module implements an interactive foreground extraction algorithm that combines:
- Expectation-Maximization (EM) for modeling foreground/background color distributions
- Tree-Reweighted Sequential Message Passing (TRW-S) for energy minimization
- Gaussian Mixture Models (GMM) for robust color modeling

The algorithm alternates between:
1. Modeling color distributions with GMM-EM
2. Performing graph-cut segmentation with TRW-S
to refine the segmentation iteratively.
"""

import argparse
import time

import numpy as np
from colordict import ColorDict
from skimage.io import imread, imsave

from src.trws import trws
from src.weights import calculate_beta, calculate_penalties


def main():
    """
    Perform foreground extraction using interactive segmentation.

    The algorithm workflow:
    1. Load image and user-provided mask (with foreground/background markers)
    2. Extract initial foreground/background pixel samples from mask
    3. Calculate smoothness parameter (beta) from image gradients
    4. Iteratively refine segmentation:
       a. Model color distributions using GMM-EM
       b. Calculate unary and pairwise energy terms
       c. Optimize energy using TRW-S message passing
       d. Update foreground/background samples based on current labeling
    5. Extract foreground object and save results
    """
    parser = argparse.ArgumentParser(
        description="Foreground extraction using EM and TRW-S algorithms."
    )
    parser.add_argument(
        "--img_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--mask_path", type=str, required=True, help="Path to the interactive mask."
    )
    parser.add_argument(
        "--gamma",
        type=int,
        required=True,
        help="Smoothness coefficient. Range: 10-100.",
    )
    parser.add_argument(
        "--n_bg",
        type=int,
        required=True,
        help="Number of Gaussian components for background GMM.",
    )
    parser.add_argument(
        "--n_fg",
        type=int,
        required=True,
        help="Number of Gaussian components for foreground GMM.",
    )
    parser.add_argument(
        "--bg_color",
        type=str,
        required=True,
        help="Color marking background in the mask (e.g., 'red', 'blue').",
    )
    parser.add_argument(
        "--fg_color",
        type=str,
        required=True,
        help="Color marking foreground in the mask (e.g., 'green', 'yellow').",
    )
    parser.add_argument(
        "--em_n_iter",
        type=int,
        required=True,
        help="Number of EM iterations for GMM fitting per iteration.",
    )
    parser.add_argument(
        "--trws_n_iter",
        type=int,
        required=True,
        help="Number of TRW-S message passing iterations per iteration.",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        required=True,
        help="Total number of refinement iterations (alternating EM and TRW-S).",
    )

    args = parser.parse_args()

    # Parse color names to RGB values
    color_dict = ColorDict()
    bg_color_rgb = np.array(color_dict[args.bg_color], dtype=int)
    fg_color_rgb = np.array(color_dict[args.fg_color], dtype=int)

    # Load input data
    image = imread(args.img_path).astype("float64")
    height, width, _ = image.shape
    mask = imread(args.mask_path).astype("int")

    time1 = time.perf_counter()

    # Define label set: K = {0: background, 1: foreground}
    K = np.array([0, 1])
    n_labels = len(K)

    # Extract initial pixel samples from user scribbles
    # bg: pixels marked as background in the mask
    # fg: pixels marked as foreground in the mask
    bg = image[np.all(mask == bg_color_rgb, axis=2)]
    fg = image[np.all(mask == fg_color_rgb, axis=2)]

    # Calculate beta: the variance parameter for smoothness term
    # Beta is computed from average color differences between neighboring pixels
    beta = calculate_beta(image)

    # Initial energy calculation
    # Q: unary penalties based on color models
    # g: pairwise penalties based on color gradients
    Q, g = calculate_penalties(
        image, args.gamma, beta, K, fg, bg, args.n_fg, args.n_bg, args.em_n_iter
    )

    # Initialize message array P for TRW-S
    # Shape: (height, width, 4, n_labels)
    # Dimension 2 represents 4 directions: [Left, Right, Up, Down]
    P = np.zeros((height, width, 4, n_labels))

    # Perform initial TRW-S optimization
    labelling = trws(height, width, n_labels, K, Q, g, P, args.trws_n_iter)

    # Iterative refinement: alternate between color modeling and segmentation
    for _ in range(args.n_iter - 1):
        # Update foreground/background samples based on current labeling
        bg = image[labelling == 0]
        fg = image[labelling == 1]

        # Recompute energy terms with updated color models
        Q, g = calculate_penalties(
            image, args.gamma, beta, K, fg, bg, args.n_fg, args.n_bg, args.em_n_iter
        )

        # Optimize energy with updated terms
        labelling = trws(height, width, n_labels, K, Q, g, P, args.trws_n_iter)

    # Extract foreground object: keep only pixels labeled as foreground
    extracted = np.zeros_like(image, dtype=np.uint8)
    extracted[labelling == 1] = image[labelling == 1]

    time2 = time.perf_counter()
    print(f"Total time: {time2 - time1:.2f} seconds")

    # Save results
    imsave("segmentation.png", labelling.astype(np.uint8) * 255)
    imsave("extracted.png", extracted)

    print("Segmentation saved to segmentation.png")
    print("Extracted object saved to extracted.png")


if __name__ == "__main__":
    main()
