# Statistical Pattern Recognition

Laboratory works from the university course.

## Lab 1 – Recognition of a noised string

### Description
The program converts a string to a noised image and then decodes it.
Dynamic programming algorithm for chain-structured graphical models.

### Usage
```commandline
 $ python3 lab1/decode_string.py --help
usage: decode_string.py [-h] --input_string INPUT_STRING --noise_level NOISE_LEVEL [--seed SEED]

options:
  -h, --help            show this help message and exit
  --input_string INPUT_STRING
                        input string
  --noise_level NOISE_LEVEL
                        noise level of bernoulli distribution
  --seed SEED           seed to debug
```
### Examples
```bash
python3 decode_string.py --input_string "billy herrington" --noise_level 0.35 --seed 45
```
Decoded string: "billy herrington"

| Original image                        |           Noised image           | Decoded image                     |
|---------------------------------------|:--------------------------------:|-----------------------------------|
| ![](.imgs/lab1/test1/input_image.png) | ![](.imgs/lab1/test1/noised_image.png) | ![](.imgs/lab1/test1/decoded_image.png) |

## Lab 2 - Image segmentation

### Description
The program segmentates a noised image using Min-Sum Diffusion.

### Usage
```commandline
$ python3 image_denoiser.py --help
usage: image_denoiser.py [-h] --img_path IMG_PATH --alpha ALPHA [--n_iter N_ITER] [--c C [C ...]]

Image segmentation on a noised image using diffusion.

options:
  -h, --help           show this help message and exit
  --img_path IMG_PATH  Path to the image to denoise
  --alpha ALPHA        Alpha parameter for binary penalties
  --n_iter N_ITER      Number of iterations
  --c C [C ...]        List of colors to segment
```

### Examples

```bash
python3 image_denoiser.py --img_path "test_images/map_hsv.png" --alpha 3 --n_iter 100 --c "blue lime"
```

|              Noised image              | Decoded image                              |
|:--------------------------------------:|--------------------------------------------|
| ![](labs/lab2/test_images/map_hsv.png) | ![](.imgs/lab2/test2/denoised_map_hsv.png) |


```bash
python3 image_denoiser.py --img_path "test_images/ipt.png" --alpha 1 --n_iter 100 --c "blue yellow white"
```

|            Noised image            | Decoded image                          |
|:----------------------------------:|----------------------------------------|
| ![](labs/lab2/test_images/ipt.png) | ![](.imgs/lab2/test2/denoised_ipt.png) |


## Lab 3 - Image Inpainting 

### Description
The program inpaint mask regions using Tree Reweighted Message Passing (TRW-S) algorithm.

```commandline
$ python3 image_inpainter.py --help
usage: image_inpainter.py [-h] --img_path IMG_PATH --alpha ALPHA --epsilon EPSILON --n_labels N_LABELS --n_iter N_ITER

Image inpainter using TRW-S algorithm.

options:
  -h, --help           show this help message and exit
  --img_path IMG_PATH  Path to the image.
  --alpha ALPHA        Smoothing coefficient for binary penalties.
  --epsilon EPSILON    Special parameter, which is responsible for lack of color information.
  --n_labels N_LABELS  Number of labels.
  --n_iter N_ITER      Number of iterations.
```

### Examples
```bash
python3 image_inpainter.py --img_path "test_images/mona-lisa-damaged.png" --alpha 1 --epsilon 0 --n_labels 18 --n_iter 4
```

|                 Image with marks                 | Inpainted image                     |
|:------------------------------------------------:|-------------------------------------|
| ![](labs/lab3/test_images/mona-lisa-damaged.png) | ![](.imgs/lab3/test1/inpainted.png) |


## Lab 4 - "GrabCut"
**_NOTE:_**  TRW-S as an energy minimization algorithm (instead of Min-Cut/Max-Flow algorithm)
### Interactive Foreground Extraction
#### Examples
```bash
python3 lab4/main.py image_path mask_path gamma n_bg n_fg color_bg color_fg em_n_iter trws_n_iter n_iter 

python3 lab4/main.py lab4/test_images/alpaca.jpg lab4/test_images/alpaca-segmentation.png  50 3 3 blue red 10 10 1
python3 lab4/main.py lab4/test_images/lotus.jpg lab4/test_images/lotus-segmentation.png  50 3 3 lime blue 10 10 1
```

## Setup

To run these applications, you need to have **Python3.12**.

1. Clone repo

2. Create virtual environment.
```bash
python3.12 -m venv .venv
```

3. Activate it
```bash
source .venv/bin/activate
```

4. Install requirements:
```bash
pip install -r requirements.txt
