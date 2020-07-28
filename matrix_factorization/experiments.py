import copy
import random
from timeit import default_timer as timer

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


def computeMeanVariance(error_list: list) -> tuple:
    """
    computes the mean and the variance for every single list in @error_list
    @param error_list: contains lists of errors for several fractions
    @return: tuple of mean and variance for every list
    """
    means = []
    variances = []
    for e_list in error_list:
        means.append(np.mean(e_list))
        variances.append(np.var(e_list))
    return means, variances


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
    svd_times = []
    uv_times = []
    impute_times = []
    svd_time = []
    uv_time = []
    impute_time = []
    svd_errors = []
    uv_errors = []
    impute_errors = []
    svd_error = []
    uv_error = []
    impute_error = []
    for fraction in tqdm(fractions):
        svd_time.clear()
        uv_time.clear()
        impute_time.clear()
        svd_error.clear()
        uv_error.clear()
        impute_error.clear()
        # Iterate ten times to eliminate outliers
        for _ in tqdm(range(10)):
            print("Generate Matrix")
            x = generateSquareRandomMatrix(5000)
            print(f"Delete random values{fraction}")
            x_tilde = deleteValues(x, fraction=fraction)
            start = timer()
            x_svd = iterativeSVD(x_tilde)
            svd_time.append(timer() - start)
            start = timer()
            x_uv = matrix_completion(x_tilde)
            uv_time.append(timer() - start)
            start = timer()
            x_impute = softImpute(x_tilde)
            impute_time.append(timer() - start)
            print("Start computing the errors")
            svd_error.append(computeRMSE(x, x_svd))
            uv_error.append(computeRMSE(x, x_uv))
            impute_error.append(computeRMSE(x, x_impute))
        svd_times.append(svd_time)
        uv_times.append(uv_time)
        impute_times.append(impute_time)
        svd_errors.append(svd_error)
        uv_errors.append(uv_error)
        impute_errors.append(impute_error)
    svd_time_expectation = computeMeanVariance(svd_times)[0]
    uv_time_expectation = computeMeanVariance(uv_times)[0]
    impute_time_expectation = computeMeanVariance(impute_times)[0]
    times = svd_time_expectation, uv_time_expectation, impute_time_expectation
    svd_moments = computeMeanVariance(svd_errors)
    uv_moments = computeMeanVariance(uv_errors)
    impute_moments = computeMeanVariance(impute_errors)
    moments = svd_moments[0], uv_moments[0], impute_moments[0]
    variances = svd_moments[1], uv_moments[1], impute_moments[1]
    print('Plotting')
    createPlots(times, fractions, name='Measured time')
    createPlots(moments, fractions, name="Expectation values")
    createPlots(variances, fractions, name="Variances")


if __name__ == "__main__":
    fire.Fire(startExperiment)
