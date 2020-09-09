from timeit import default_timer as timer

import fire
import numpy as np
from tqdm import tqdm

from matrix_factorization.factorization import iterativeSVD, softImpute
from utils import computeRMSE, createPlots, generateSquareRandomMatrix, deleteValues, computeMeanVariance


def startExperiment():
    fractions = np.arange(0.1, 1, 0.1)
    svd_times = []
    # uv_times = []
    impute_times = []
    svd_time = []
    uv_time = []
    impute_time = []
    svd_errors = []
    # uv_errors = []
    impute_errors = []
    svd_error = []
    uv_error = []
    impute_error = []
    for fraction in tqdm(fractions):
        svd_time.clear()
        uv_time.clear()
        impute_time.clear()
        svd_error.clear()
        # uv_error.clear()
        impute_error.clear()
        for _ in tqdm(range(5)):
            print("Generate Matrix")
            x = generateSquareRandomMatrix(5000)
            print(f"Delete random values{fraction}")
            x_tilde = deleteValues(x, fraction=fraction)
            start = timer()
            x_svd = iterativeSVD(x_tilde)
            diff = (timer() - start) // 60
            svd_time.append(diff)
            # start = timer()
            # x_uv = matrix_completion(x_tilde)
            # diff = (timer() - start) // 60
            # uv_time.append(diff)
            start = timer()
            x_impute = softImpute(x_tilde)
            diff = (timer() - start) // 60
            impute_time.append(diff)
            print("Start computing the errors")
            svd_error.append(computeRMSE(x, x_svd))
            # uv_error.append(computeRMSE(x, x_uv))
            impute_error.append(computeRMSE(x, x_impute))
        svd_times.append(svd_time)
        # uv_times.append(uv_time)
        impute_times.append(impute_time)
        svd_errors.append(svd_error)
        # uv_errors.append(uv_error)
        impute_errors.append(impute_error)
    svd_time_expectation = computeMeanVariance(svd_times)[0]
    # uv_time_expectation = computeMeanVariance(uv_times)[0]
    impute_time_expectation = computeMeanVariance(impute_times)[0]
    svd_moments = computeMeanVariance(svd_errors)
    # uv_moments = computeMeanVariance(uv_errors)
    impute_moments = computeMeanVariance(impute_errors)
    times = svd_time_expectation, impute_time_expectation,  # uv_time_expectation
    moments = svd_moments[0], impute_moments[0],  # uv_moments[0]
    variances = svd_moments[1], impute_moments[1],  # uv_moments[1]
    print('Plotting')
    createPlots(times, fractions, name='Time5000', title='Time over fraction of approximated kernel values',
                xlabel='Fraction of approximated values', ylabel='Time in minutes', mf=True)
    createPlots(moments, fractions, name="Error5000", title='Mean error over fraction of approximated kernel values'
                , xlabel='Fraction of approximated values', ylabel='RMSE', mf=True)
    createPlots(variances, fractions, name="Variances5000",
                title='Variances of the error over fraction of approximated kernel values',
                xlabel='Fraction of approximated values', ylabel='Variance of RMSE', mf=True)


if __name__ == "__main__":
    fire.Fire(startExperiment)
