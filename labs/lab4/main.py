from functionality import *
import matplotlib.pyplot as plt
from skimage.io import imread
from colordict import ColorDict
import sys
import time

# image mask gamma n_bg n_fg color_bg color_fg em_n_iter trws_n_iter n_iter 
def main():

    '''
    Parameters
        argv[1]: str 
            image: image_path
        argv[2]: str
            mask: mask_path
        argv[3]: str
            gamma: normalization coefficient
        argv[4]: str
            n_bg: number of components for background modelling
        argv[5]: str
            n_fg: number of components for foreground modelling
        argv[6]: str
            color_bg: rgb color of background
        argv[7]: str
            collor_fg: rgb color of foreground
        argv[8]: str
            em_n_iter: number of iterations for EM algorithm
        argv[9]: str
            trws_n_iter: number of iterations for TRW-S algorithm
        argv[10]: str
            n_iter: total number of iterations
    '''

    # initialise parameters
    image_path, mask_path, gamma, n_bg, n_fg, color_bg, color_fg, em_n_iter, trws_n_iter, n_iter  = sys.argv[1:11]
    gamma = int(gamma)
    n_bg, n_fg = int(n_bg), int(n_fg)
    em_n_iter, trws_n_iter, n_iter = int(em_n_iter), int(trws_n_iter), int(n_iter)

    # colors - dict of rgb colors codes, to see all colors, please use print(ColorDict())
    colors = ColorDict()
    color_bg, color_fg = np.array(colors[color_bg],dtype=int), np.array(colors[color_fg],dtype=int) #[0,255,0], [0,0,255]

    image = imread(image_path).astype("float64")
    mask = imread(mask_path).astype("int")
    # initialize parameters
    params = {'image': image,
              'mask': mask,
              'color_bg': color_bg,
              'color_fg': color_fg,
              'trws_n_iter': trws_n_iter,
              'em_n_iter': em_n_iter,
              'n_iter': n_iter,
              'gamma': gamma,
              'n_bg': n_bg,
              'n_fg': n_fg
             }

    a = time.time()

    labelling = process(params)
    # highlight detected object
    object_image = np.zeros_like(image,dtype= 'uint8')
    object_image[labelling==1] = image[labelling==1]

    print("total time", time.time()-a)

    plt.figure(figsize = (20,20))

    plt.subplot(1,3,1), plt.imshow(image.astype('uint8')), plt.axis('off'), plt.title("input image")

    plt.subplot(1,3,2), plt.imshow(labelling), plt.axis('off'), plt.title("segmentation")
    plt.subplot(1,3,3), plt.imshow(object_image), plt.axis('off'), plt.title("detected object")

    plt.show()     



if __name__ == "__main__":
    main()

