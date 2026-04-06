import argparse
import json
import string
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from PIL import Image


class StringImageDecoder:
    """Converts strings to images, applies noise, and decodes using dynamic programming.

    Each alphabet character is a grayscale image with shape (height, 27).
    Decoding uses bigram probabilities to find the most likely string.
    """

    def __init__(
        self,
        input_string: str,
        noise: float,
        alphabet_path: str,
        frequencies_path: str,
        seed: int,
    ):
        """Initialize the decoder with alphabet images and bigram frequencies.

        Args:
            input_string: Text to encode and decode
            noise: Probability of bit flip (0-1)
            alphabet_path: Directory containing character images
            frequencies_path: JSON file with bigram frequencies
            seed: Random seed for reproducibility
        """
        self.input_string = input_string
        self.noise = noise
        self.alphabet_path = alphabet_path
        self.frequencies_path = frequencies_path

        assert 0 <= self.noise <= 1, "Noise level should be in range [0, 1]."

        self.alphabet: list[str] = list(string.ascii_lowercase + " ")

        with open(self.frequencies_path) as json_file:
            self.frequencies_dict = json.load(json_file)

        self.alphabet_dict: dict[str, np.ndarray] = self._read_alphabet_folder()
        self.alphabet_imgs: list[np.ndarray] = list(self.alphabet_dict.values())

        self.input_string_im = self.string_to_image(self.input_string)

        np.random.seed(seed)
        self.noised_im = self._add_binomial_noise()
        self.bigram_probs = self._calculate_bigram_log_probs()

        self.decoded_string = ""
        self.decoded_image = np.zeros_like(self.noised_im)

        print("Input string converted to image. Bigrams calculated.")

    def _read_alphabet_folder(self) -> dict[str, np.ndarray]:
        """Load character images from the disk."""
        alphabet_dict = {}
        for letter in self.alphabet[:-1] + ["space"]:
            alphabet_dict[letter] = np.array(
                Image.open(f"{self.alphabet_path}/{letter}.png"), dtype=int
            )
        alphabet_dict[" "] = alphabet_dict.pop("space")
        return alphabet_dict

    def string_to_image(self, string_to_convert: str) -> np.ndarray:
        """Concatenate character images horizontally."""
        char_arrays = [self.alphabet_dict[letter] for letter in string_to_convert]
        return np.concatenate(char_arrays, axis=1)

    def _add_binomial_noise(self) -> np.ndarray:
        """Flip each pixel with probability = noise level."""
        noise = np.random.binomial(n=1, p=self.noise, size=self.input_string_im.shape)
        return noise ^ self.input_string_im

    def _calculate_bigram_log_probs(self) -> np.ndarray:
        """Compute log probabilities for all character bigrams.

        Returns:
            Matrix of shape (27, 27) where entry [i,j] is log P(j|i).
            Impossible bigrams have -inf probability.
        """
        freq_matrix = np.zeros((len(self.alphabet), len(self.alphabet)), dtype=int)

        for i, letter_i in enumerate(self.alphabet):
            for j, letter_j in enumerate(self.alphabet):
                bigram = letter_i + letter_j
                if bigram in self.frequencies_dict:
                    freq_matrix[i][j] = self.frequencies_dict[bigram]

        # Normalize rows to get conditional probabilities
        bigram_probs = (freq_matrix.T / freq_matrix.sum(axis=1)).T

        # Convert to log space, set zero probabilities to -inf
        bigram_probs = np.log(
            bigram_probs,
            out=np.full_like(bigram_probs, -np.inf),
            where=(bigram_probs != 0),
        )

        return bigram_probs

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _compute_penalties(
        img: np.ndarray,
        letters: list[np.ndarray],
        penalties: np.ndarray,
        prev_positions: np.ndarray,
        next_positions: np.ndarray,
        p: float,
        bigram_log_probs: np.ndarray,
    ):
        """JIT-compiled dynamic programming step for efficient penalty calculation."""
        for prev_pos in prev_positions:
            for i, letter in enumerate(letters):
                next_pos = letter.shape[1] + prev_pos

                if next_pos <= img.shape[1]:
                    next_positions = np.append(next_positions, next_pos)

                    # Match letter template with image slice
                    img_slice = img[:, prev_pos:next_pos]

                    # Log-likelihood: P(observed | template, noise)
                    match_penalty = np.sum(
                        (img_slice ^ letter) * np.log(p)
                        + (1 ^ img_slice ^ letter) * np.log(1 - p)
                    )

                    # Add bigram probability and previous best score
                    total_penalty = (
                        match_penalty + bigram_log_probs[:, i] + penalties[prev_pos]
                    )

                    penalties[next_pos, i] = max(
                        total_penalty.max(), penalties[next_pos, i]
                    )

        return penalties, next_positions

    def decode_noised_image(self) -> None:
        """Decode noisy image using Viterbi-like dynamic programming."""
        img = self.noised_im.copy()

        p_log = np.log(self.noise)
        p1_log = np.log(1 - self.noise)

        # penalties[position, char_idx] = best log-probability ending at position with char_idx
        penalties = np.full([img.shape[1] + 1, len(self.alphabet_imgs)], -np.inf)

        letter_widths = [letter.shape[1] for letter in self.alphabet_imgs]
        min_width = min(letter_widths)

        # Initialize the first character
        prev_positions = np.array([], dtype=int)
        for i, letter_img in enumerate(self.alphabet_imgs):
            width = letter_img.shape[1]
            if width <= img.shape[1]:
                prev_positions = np.append(prev_positions, width)
                img_slice = img[:, :width]

                # Compute log-likelihood for the first character (no previous bigram)
                log_likelihood = np.sum(
                    (img_slice ^ letter_img) * p_log
                    + np.logical_xor(1, img_slice ^ letter_img) * p1_log
                )
                penalties[width, i] = log_likelihood + self.bigram_probs[-1, i]

        prev_positions = list(set(prev_positions))

        # Forward pass: fill penalty table
        while min(prev_positions) + min_width <= img.shape[1]:
            next_positions = np.array([], dtype=int)
            penalties, next_positions = self._compute_penalties(
                img,
                self.alphabet_imgs,
                penalties,
                prev_positions,
                next_positions,
                self.noise,
                self.bigram_probs,
            )
            prev_positions = np.array(list(set(next_positions)))

        # Backward pass: reconstruct the best path
        penalties = penalties[::-1]
        decoded_chars = []

        last_char_idx = np.argmax(penalties[0])
        decoded_chars.append(self.alphabet[last_char_idx])

        pos = letter_widths[last_char_idx]
        while pos <= penalties.shape[0] - 2:
            # Find the best previous character given current character
            prev_char_idx = np.argmax(
                penalties[pos] + self.bigram_probs[:, last_char_idx]
            )
            decoded_chars.append(self.alphabet[prev_char_idx])

            pos += letter_widths[prev_char_idx]
            last_char_idx = prev_char_idx

        self.decoded_string = "".join(reversed(decoded_chars))
        self.decoded_image = self.string_to_image(self.decoded_string)


def main():
    parser = argparse.ArgumentParser(
        description="Decode noisy string images using dynamic programming"
    )
    parser.add_argument(
        "--input_string", type=str, required=True, help="String to encode and decode"
    )
    parser.add_argument(
        "--noise_level", type=float, required=True, help="Bit flip probability [0-1]"
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()

    decoder = StringImageDecoder(
        input_string=args.input_string,
        noise=args.noise_level,
        alphabet_path="alphabet",
        frequencies_path="frequencies.json",
        seed=args.seed,
    )
    t1 = perf_counter()
    decoder.decode_noised_image()
    t2 = perf_counter()

    print(f"Input string:   {decoder.input_string}")
    print(f"Decoded string: {decoder.decoded_string}")
    print(f"Time:           {t2 - t1:.2f} seconds")

    plt.imsave(
        "../../.imgs/lab1/test1/input_image.png", decoder.input_string_im, cmap="binary"
    )
    plt.imsave(
        "../../.imgs/lab1/test1/noised_image.png", decoder.noised_im, cmap="binary"
    )
    plt.imsave(
        "../../.imgs/lab1/test1/decoded_image.png", decoder.decoded_image, cmap="binary"
    )

if __name__ == "__main__":
    main()
