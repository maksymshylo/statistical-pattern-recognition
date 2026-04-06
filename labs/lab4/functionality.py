import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import sys
import time
from sklearn import mixture
from numba import njit
from trws_algorithm import *
from gmm_algorithm import *
from types import SimpleNamespace



@njit(fastmath=True)
def calculate_q(image,n_bg,n_fg,bg_params,fg_params,Q):

    '''
    Parameters
        image: ndarray
            input image as array
        n_bg: int
            number of components for background modelling
        n_fg: int
            number of components for foreground modelling
        bg_params: tuple
            background disribution params
        fg_params: tuple
            foreground disribution params
        Q: ndarray
            array of unary penalties

    Returns
        Q: ndarray
            array of unary penalties
    updates unary penalties according to disribution
    '''

    height,width,_ = image.shape
    # initialize disribution parameters for background and foreground 
    bg_det_cov,bg_weights, bg_inv_cov,bg_means = bg_params
    fg_det_cov,fg_weights, fg_inv_cov,fg_means = fg_params

    c = (2*np.pi)**(-5/2)
    # for each  pixel calculate q
    for i in range(height):
        for j in range(width):
            # calculate p(x_t,s_t)
            b_prod = np.zeros((n_bg))
            f_prod = np.zeros((n_bg))
            # for background
            for n in range(n_bg):
                bg_cond_prob = np.exp(-(1/2)*(image[i,j,:]-bg_means[n]).T@bg_inv_cov[n]@(image[i,j,:]-bg_means[n]))
                b_prod[n] = bg_weights[n]*bg_det_cov[n]*bg_cond_prob
            # for foreground
            for n in range(n_fg):
                fg_cond_prob = np.exp(-(1/2)*(image[i,j,:]-fg_means[n]).T@fg_inv_cov[n]@(image[i,j,:]-fg_means[n]))            
                f_prod[n] = fg_weights[n]*fg_det_cov[n]*fg_cond_prob
            
            Q[i,j,1] = np.log(np.sum(c*f_prod)) # 1 - foreground
            Q[i,j,0] = np.log(np.sum(c*b_prod)) # 0 - background
    return Q




@njit(fastmath=True)
def calculate_beta(image):

    '''
    Parameters
        image: ndarray
            input image as array
    Returns
        beta: float
            variance for binary penalties calculation
    calculates beta

    '''

    height,width,_ = image.shape
    beta = 0.0
    # initializing tau - number of edges in a grid graph
    tau = 2*height*width - height - width
    # for each pixel in image
    for i in range(height):
        for j in range(width):
            # calculating right and bottom neighbours
            neighbours = get_right_down(height,width,i,j)
            for [ni,nj] in neighbours:
                beta += np.linalg.norm(image[i,j,:] - image[ni,nj,:])**2
    beta /= tau
    return beta


@njit(fastmath=True)
def calculate_g(image,gamma,beta,g):

    '''
    Parameters
        image: ndarray
            input image as array
        gamma: int
            normalization parameter (usually 10 <= gamma <= 100)
        beta: float
            variance for binary penalties calculation
        g: ndarray
            binary penalties
    Returns
        g: ndarray
            updated binary penalties
    updates binary penalties according to disribution
    '''

    height,width,_ = image.shape
    # for each pixel in image
    for i in range(height):
        for j in range(width):
            nbs, _ , nbs_indices = get_neighbours(height, width,i,j)
            # for each neighbour
            for n,[n_i,n_j] in enumerate(nbs):
                g[i,j,nbs_indices[n],0,1] = -gamma*np.exp(-np.linalg.norm(image[i,j,:]-image[n_i,n_j,:])**2/(2*beta))
                g[i,j,nbs_indices[n],1,0] = -gamma*np.exp(-np.linalg.norm(image[i,j,:]-image[n_i,n_j,:])**2/(2*beta))
    return g        



def calculate_penalties(image,gamma,beta,K,fg,bg,n_fg,n_bg,em_n_iter):

    '''
    Parameters
        image: ndarray
            input image as array
        gamma: int
            normalization parameter (usually 10 <= gamma <= 100)
        K: ndarray
            array of labels
        fg: ndarray
            foreground pixels sample
        bg: ndarray
            background pixels sample
        n_fg: int
            number of components for foreground modelling
        n_bg: int
            number of components for background modelling
        em_n_iter: int
            number of iterations for EM algorithm

    Returns
        Q: ndarray
            updated unary penalties
        g: ndarray
            updated binary penalties
    updates unary and binary penalties according to disribution
    '''

    height, width, _ = image.shape
    # apply EM algorithm for background and foreground pixels
    fg_means,fg_sigma,_,fg_weights =  gmm(fg, n_fg, n_iter = em_n_iter)
    bg_means,bg_sigma,_,bg_weights =  gmm(bg, n_bg, n_iter = em_n_iter)

    # initialize constants for distributions
    fg_det_cov = np.linalg.det(fg_sigma)**(-1/2)
    fg_inv_cov = np.linalg.inv(fg_sigma)

    bg_det_cov = np.linalg.det(bg_sigma)**(-1/2)
    bg_inv_cov = np.linalg.inv(bg_sigma) 

    bg_params = (bg_det_cov,bg_weights, bg_inv_cov,bg_means)
    fg_params = (fg_det_cov,fg_weights, fg_inv_cov,fg_means)
    # update Q
    Q = np.empty((height,width,len(K)))
    Q =  calculate_q(image,n_bg,n_fg,bg_params, fg_params,Q)

    # update g
    g = np.zeros((height,width,4,len(K),len(K)))
    g = calculate_g(image,gamma,beta,g)

    return Q,g

def process(params):

    '''
    Parameters
        params: dict
            image: ndarray
                input image as array
            mask: ndarray
                input mask with partial segmentation
            color_bg: ndarray
                rgb color of background
            color_fg: ndarray
                rgb color of foreground
            trws_n_iter: int
                number of iterations for TRW-S algorithm
            em_n_iter: int 
                number of iterations for EM algorithm
            n_iter: int 
                total number of iterations
            gamma: int
                normalization parameter (usually 10 <= gamma <= 100)
            n_bg: int
                number of components for background modelling
            n_fg: int
                number of components for foreground modelling
    Returns
        labelling: ndarray
            optimal labelling after full iteration of GRAB CUT algorithm
       
    best labelling after one iteration of Grab Cut algorithm
    '''
    # unpack parameters from dictionary
    params = SimpleNamespace(**params)
    # define set if labels
    K = np.array([0,1])
    # initial sample pixels for background and foreground
    bg = params.image[np.all(params.mask==params.color_bg,axis=2)]
    fg = params.image[np.all(params.mask==params.color_fg,axis=2)]
    # calculate beta only one time
    # calculate penalties
    beta = calculate_beta(params.image)
    Q,g = calculate_penalties(params.image, params.gamma, beta, K, fg, bg, params.n_fg, params.n_bg, params.em_n_iter)
    # calculate best labelling for given Q and g
    labelling = optimal_labelling(Q,g,K,params.trws_n_iter)
    # for number of iterations repeat
    for _ in range(params.n_iter-1):
        bg = params.image[labelling==0]
        fg = params.image[labelling==1]
        Q,g = calculate_penalties(params.image, params.gamma, beta, K, fg, bg, params.n_fg, params.n_bg, params.em_n_iter)
        labelling = optimal_labelling(Q,g,K,params.trws_n_iter)
    # best labelling after n_iter iterations of Grab Cut algorithm
    return labelling
