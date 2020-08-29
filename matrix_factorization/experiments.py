from timeit import default_timer as timer

import fire
import numpy as np
from tqdm import tqdm

from matrix_factorization.factorization import iterativeSVD, softImpute, matrix_completion
from utils.utils import computeRMSE, createPlots, generateSquareRandomMatrix, deleteValues, computeMeanVariance


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
