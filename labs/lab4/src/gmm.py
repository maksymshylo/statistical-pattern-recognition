import numpy as np
from scipy.stats import multivariate_normal

np.random.seed(42)


def predict_proba(X: np.ndarray, n_components: int, mu, sigma, phi) -> np.ndarray:
    """
    Calculate probabilities for each component.

    Args:
        X: Multivariate distribution points
        n_components: Number of components (clusters)
        mu: List of means for each component
        sigma: List of covariances for each component
        phi: Array of  apriori probabilities (p(k))

    Returns:
        Array of probabilities for each component
    """

    n, _ = X.shape
    # initialize likelihood array
    likelihood = np.zeros((n, n_components))
    for i in range(n_components):
        # calculate probability
        distribution = multivariate_normal(mean=mu[i], cov=sigma[i])
        likelihood[:, i] = distribution.pdf(X)

    numerator = likelihood * phi
    weights = numerator / np.sum(numerator, axis=1)[:, np.newaxis]
    return weights


def em(X, n_components, mu, sigma, phi) -> tuple:
    """
    Expectation-Maximization algorithm.

    Args:
        X: Multivariate distribution points.
        n_components: Number of components (clusters).
        mu: List of means for each component.
        sigma: List of covariances for each component.
        weights: Array of probabilities for each pixel.
        phi: Array of apriori probabilities (p(k)).

    Returns:
        Updated args.
    """

    # expectation step
    weights = predict_proba(X, n_components, mu, sigma, phi)
    phi = np.mean(weights, axis=0)

    # maximization step
    for i in range(n_components):
        weight = weights[:, [i]]
        total_weight = weight.sum()
        mu[i] = np.sum(X * weight, axis=0) / total_weight
        sigma[i] = np.cov(X.T, aweights=(weight / total_weight).flatten(), bias=True)
    return mu, sigma, weights, phi


def gmm(X: np.ndarray, n_components: int, n_iter: int = 100) -> tuple:
    """
    Perform n_iter iterations of EM algorithm

    Args:

        X: Multivariate distribution points.
        n_components: Number of components (clusters).
        n_iter: Number of iterations.

    Returns:
        mu,sigma, weights, phi at n_iter iteration

    """
    if X.size == 0:
        raise Exception("X sample is empty")
    if n_components <= 1:
        raise Exception("n_components can not be less than 1")

    # initialization
    n, m = X.shape
    phi = np.full(n_components, 1 / n_components)
    weights = np.full(X.shape, 1 / n_components)
    random_mu = np.random.randint(0, n, n_components)
    mu = list(X[random_mu, :])
    sigma = [np.cov(X.T)] * n_components

    # EM
    for _ in range(n_iter):
        mu, sigma, weights, phi = em(X, n_components, mu, sigma, phi)

    return mu, sigma, weights, phi
