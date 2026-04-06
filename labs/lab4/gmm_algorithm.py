from scipy.stats import multivariate_normal
import numpy as np

def predict_proba(X,n_components,mu,sigma,phi):

    '''
    Parameters
        X: ndarray
           array of multivariate distribution points
        n_components: int
            number of components(clusters) 
        mu: list
            list of means for each component
        sigma: list
            list of covariances for each component
        phi: ndarray
            array of apriori probabilities(p(k))
    Returns
        weights: ndarray
            array of probabilities for each pixel
    calculates weights
    '''

    n,_ = X.shape
    # initialize likelihood array
    likelihood = np.zeros( (n, n_components) )
    for i in range(n_components):
        # calculate probability
        distribution = multivariate_normal(mean=mu[i], cov=sigma[i])
        likelihood[:,i] = distribution.pdf(X)
    
    numerator = likelihood * phi
    weights = numerator / np.sum(numerator,axis=1)[:, np.newaxis]
    return weights

def em(X,n_components,mu,sigma,weights,phi):

    '''
    Parameters
        X: ndarray
           array of multivariate distribution points
        n_components: int
            number of components(clusters) 
        mu: list
            list of means for each component
        sigma: list
            list of covariances for each component
        weights: ndarray
            array of probabilities for each pixel
        phi: ndarray
            array of apriori probabilities(p(k))
    Returns
        updated mu,sigma,weights,phi
    updates mu,sigma,weights,phi
    '''

    # expectation step
    weights =  predict_proba(X,n_components,mu,sigma,phi)
    phi = np.mean(weights,axis=0)

    # maximization step
    for i in range(n_components):
        weight =  weights[:, [i]]
        total_weight = weight.sum()
        mu[i] = np.sum(X * weight,axis=0) / total_weight
        sigma[i] = np.cov(X.T, aweights=(weight/total_weight).flatten(), bias=True)
    return mu,sigma,weights,phi

def gmm(X,n_components, n_iter = 100):

    '''
    Parameters
        X: ndarray
           array of multivariate distribution points
        n_components: int
            number of components(clusters) 
        n_iter: int
            number of iterations
    Returns
        mu,sigma,weights,phi at n_iter iteration
    initialize parameters and run n_iter iterations of EM algorithm
    examples:
    >>> gmm(np.array([]),4,100)
    Traceback (most recent call last):
    ...
    Exception: X sample is empty
    >>> gmm(np.array([1]),0,100)
    Traceback (most recent call last):
    ...
    Exception: n_components can not be less than 1
    '''
    if X.size == 0:
        raise Exception('X sample is empty')
    if n_components <= 1:
        raise Exception('n_components can not be less than 1')

    # initialization
    n,m = X.shape
    phi = np.full(n_components, 1/n_components)
    weights = np.full(X.shape,  1/n_components)
    random_mu = np.random.randint(0, n, n_components)
    mu = list(X[random_mu,:])
    sigma = [np.cov(X.T)]*n_components

    # EM
    for _ in range(n_iter):
        mu,sigma,weights,phi = em(X,n_components,mu,sigma,weights,phi)

    return mu,sigma,weights,phi