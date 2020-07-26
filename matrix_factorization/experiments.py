import copy
import random

import fire
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from exp_mnist_resnet.classify_gp import computeRMSE
from matrix_factorization.factorization import iterativeSVD, softImpute, matrix_completion


def generateSquareRandomMatrix(columns: int) -> np.ndarray:
    """
    Generate a random squared matrix with values between zero and 1 with the
    shape of (columns, columns)
    @param columns: amount of columns and rows (because it is squared)
    @return: random generated squared matrix
    """
    x = np.random.rand(columns, columns)
    return x


def deleteValues(x: np.ndarray, fraction: float) -> np.ndarray:
    """
    Takes a matrix and a fraction as input and returns a matrix
    where the fraction of elements are randomly set to Nan
    @param x: original matrix from which values are set to Nan
    @param fraction: percentage of all values of x is set  to NaN
    @return: perturbed matrix with NaNs
    """
    nan_x = copy.deepcopy(x)
    prop = int(fraction * x.size)
    mask = random.sample(range(x.size), prop)
    np.put(nan_x, mask, np.nan)
    return nan_x


def computeMeanVariance(error_list: list) -> list:
    """
    computes the mean and the variance for every single list in @error_list
    @param error_list: contains lists of errors for several fractions
    @return: tuple of mean and variance for every list
    """
    result = []
    for e_list in error_list:
        mean = np.mean(e_list)
        variance = np.var(e_list)
        result.append(tuple((mean, variance)))
    return result


def createPlots(moments, fractions, name):
    """
    Creates plots for the given moment data on y and fractions data on x with name as title
    @param moments: contains expected values or variances of the errors
    of the matrix approximation methods on the y-axis
    @param fractions: contains the values for the x-axis
    @param name: sets the title
    @return: None
    """
    plt.figure()
    plt.title(f"{name} values of the errors over the percentage of data which should be approximated")
    plt.plot(fractions, moments[0], label='svd iteration')
    plt.plot(fractions, moments[1], label='matrix factorization')
    plt.plot(fractions, moments[2], label='soft impute')
    plt.legend()
    plt.savefig(f'./plots/{name}.svg')


def startExperiment():
    fractions = np.arange(0.1, 1, 0.1)
    svd_errors = []
    uv_errors = []
    impute_errors = []
    svd_error = []
    uv_error = []
    impute_error = []
    for fraction in tqdm(fractions):
        svd_error.clear()
        uv_error.clear()
        impute_error.clear()
        # Iterate ten times to eliminate outliers
        for _ in tqdm(range(10)):
            print("Generate Matrix")
            x = generateSquareRandomMatrix(5000)
            print("Delete random values")
            x_tilde = deleteValues(x, fraction=fraction)
            x_svd = iterativeSVD(x_tilde)
            x_uv = matrix_completion(x_tilde)
            x_impute = softImpute(x_tilde)
            print("Start computing the errors")
            svd_error.append(computeRMSE(x, x_svd))
            uv_error.append(computeRMSE(x, x_uv))
            impute_error.append(computeRMSE(x, x_impute))
        svd_errors.append(svd_error)
        uv_errors.append(uv_error)
        impute_errors.append(impute_error)
    svd_moments = computeMeanVariance(svd_errors)
    uv_moments = computeMeanVariance(uv_errors)
    impute_moments = computeMeanVariance(impute_errors)
    moments = svd_moments[0], uv_moments[0], impute_moments[0]
    variances = svd_moments[1], uv_moments[1], impute_moments[1]
    createPlots(moments, fractions, name="Expectation values")
    createPlots(variances, fractions, name="Variances")


if __name__ == "__main__":
    fire.Fire(startExperiment)
