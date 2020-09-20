from timeit import default_timer as timer

import fire
import numpy as np
from fancyimpute import IterativeSVD, SoftImpute, MatrixFactorization
from tqdm import tqdm

from utils import createPlots, generateSquareRandomMatrix, deleteValues, computeMeanVariance, \
    computeRelativeRMSE


def measureTime(Xpert, mode):
    if mode == "svd":
        solver = IterativeSVD()
    elif mode == "mf":
        solver = MatrixFactorization(epochs=1000, min_improvement=0.01)
    elif mode == "soft":
        solver = SoftImpute()
    else:
        print("Mode chosen wrongly")
        return -1
    start = timer()
    approx = solver.fit_transform(Xpert)
    end = timer()
    diff = end - start
    minutes = diff // 60
    return approx, minutes


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
        for _ in tqdm(range(3)):
            print("Generate Matrix")
            x = generateSquareRandomMatrix(5000)
            print(f"Delete random values{fraction}")
            x_tilde = deleteValues(x, fraction=fraction)
            approx, time = measureTime(x_tilde, 'svd')
            print(time)
            svd_error.append(computeRelativeRMSE(x, approx, fraction))
            svd_time.append(time)
            approx, time = measureTime(x_tilde, 'mf')
            print(time)
            uv_error.append(computeRelativeRMSE(x, approx, fraction))
            uv_time.append(time)
            approx, time = measureTime(x_tilde, 'soft')
            print(time)
            impute_time.append(time)
            impute_error.append(computeRelativeRMSE(x, approx, fraction))
        svd_times.append(list(svd_time[:]))
        uv_times.append(list(uv_time[:]))
        impute_times.append(list(impute_time[:]))
        svd_errors.append(list(svd_error[:]))
        uv_errors.append(list(uv_error[:]))
        impute_errors.append(list(impute_error[:]))
    svd_time_expectation = computeMeanVariance(svd_times)[0]
    uv_time_expectation = computeMeanVariance(uv_times)[0]
    impute_time_expectation = computeMeanVariance(impute_times)[0]
    svd_moments = computeMeanVariance(svd_errors)
    uv_moments = computeMeanVariance(uv_errors)
    impute_moments = computeMeanVariance(impute_errors)
    times = svd_time_expectation, impute_time_expectation, uv_time_expectation
    moments = svd_moments[0], impute_moments[0], uv_moments[0]
    variances = svd_moments[1], impute_moments[1], uv_moments[1]
    # variances = uv_moments[1]
    # with h5py.File("mf.h5", 'w') as f:
    #     f.create_dataset("times", data=times)
    #     f.create_dataset("error_moments", data=uv_moments)
    #     f.close()
    print('Plotting')
    createPlots(times, fractions, name='AllTime5000', title='Time over fraction of approximated kernel values',
                xlabel='Fraction of approximated values', ylabel='Time in minutes', mf=True)
    createPlots(moments, fractions, name="AllError5000", title='Relative mean'
                                                               ' error over fraction of approximated kernel values'
                , xlabel='Fraction of approximated values', ylabel='Relative RMSE', mf=True)
    createPlots(variances, fractions, name="AllVariance5000",
                title='Variances of the relative error over fraction of approximated kernel values',
                xlabel='Fraction of approximated values', ylabel='Variance of relative RMSE', mf=True)


if __name__ == "__main__":
    fire.Fire(startExperiment)
