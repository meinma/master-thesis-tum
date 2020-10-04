from timeit import default_timer as timer

import fire
import numpy as np
from fancyimpute import IterativeSVD, SoftImpute, MatrixFactorization
from tqdm import tqdm

from utils import createPlots, generateSquareRandomMatrix, deleteValues, computeMeanVariance, \
    computeRelativeRMSE, computeErrors

ORIGINAL_PATH = './matrix_factorization/PERT.h5'
MF_PATH = './matrix_factorization/MF.h5'
ITER_PATH = './matrix_factorization/SVD.h5'
SOFT_PATH = './matrix_factorization/SOFT.h5'


def measureTime(Xpert, mode):
    if mode == "svd":
        solver = IterativeSVD()
    elif mode == "mf":
        solver = MatrixFactorization(epochs=1000, min_improvement=0.007)
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
    svd_mins = []
    svd_min = []
    svd_maxs = []
    svd_max = []
    svd_medians = []
    svd_median = []
    soft_mins = []
    soft_min = []
    soft_maxs = []
    soft_max = []
    soft_medians = []
    soft_median = []
    mf_mins = []
    mf_min = []
    mf_maxs = []
    mf_max = []
    mf_medians = []
    mf_median = []
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
        mf_max.clear()
        mf_min.clear()
        mf_median.clear()
        svd_max.clear()
        svd_min.clear()
        svd_median.clear()
        soft_max.clear()
        soft_min.clear()
        soft_median.clear()
        svd_time.clear()
        uv_time.clear()
        impute_time.clear()
        svd_error.clear()
        uv_error.clear()
        impute_error.clear()
        for _ in tqdm(range(3)):
            print("Generate Matrix")
            x = generateSquareRandomMatrix(5000)
            print(f"Delete random values {fraction}")
            x_tilde = deleteValues(x, fraction=fraction)
            approx, time = measureTime(x_tilde, 'svd')
            svd_time.append(time)
            svd_error.append(computeRelativeRMSE(x, approx, fraction))
            error = computeErrors(x, approx)
            svd_min.append(error.min_error)
            svd_max.append(error.max_error)
            svd_median.append(error.median_error)
            print(time)
            approx, time = measureTime(x_tilde, 'mf')
            print(time)
            uv_error.append(computeRelativeRMSE(x, approx, fraction))
            uv_time.append(time)
            error = computeErrors(x, approx)
            mf_min.append(error.min_error)
            mf_max.append(error.max_error)
            mf_median.append(error.median_error)
            approx, time = measureTime(x_tilde, 'soft')
            impute_time.append(time)
            impute_error.append(computeRelativeRMSE(x, approx, fraction))
            error = computeErrors(x, approx)
            soft_min.append(error.min_error)
            soft_max.append(error.max_error)
            soft_median.append(error.median_error)
            print(time)
        svd_maxs.append(list(svd_max[:]))
        svd_mins.append(list(svd_min[:]))
        svd_medians.append(list(svd_median[:]))
        soft_maxs.append(list(soft_max[:]))
        soft_mins.append(list(soft_min[:]))
        soft_medians.append(list(soft_median[:]))
        mf_maxs.append(list(mf_max[:]))
        mf_mins.append(list(mf_min[:]))
        mf_medians.append(list(mf_median[:]))
        svd_times.append(list(svd_time[:]))
        uv_times.append(list(uv_time[:]))
        impute_times.append(list(impute_time[:]))
        svd_errors.append(list(svd_error[:]))
        uv_errors.append(list(uv_error[:]))
        impute_errors.append(list(impute_error[:]))
    svd_min_expectation = computeMeanVariance(svd_mins)[0]
    svd_max_expecatation = computeMeanVariance(svd_maxs)[0]
    svd_median_expectation = computeMeanVariance(svd_medians)[0]
    mf_min_expectation = computeMeanVariance(mf_mins)[0]
    mf_max_expecatation = computeMeanVariance(mf_maxs)[0]
    mf_median_expectation = computeMeanVariance(mf_medians)[0]
    soft_min_expectation = computeMeanVariance(soft_mins)[0]
    soft_max_expectation = computeMeanVariance(soft_maxs)[0]
    soft_median_expectation = computeMeanVariance(soft_medians)[0]
    svd_time_expectation = computeMeanVariance(svd_times)[0]
    uv_time_expectation = computeMeanVariance(uv_times)[0]
    impute_time_expectation = computeMeanVariance(impute_times)[0]
    svd_moments = computeMeanVariance(svd_errors)
    uv_moments = computeMeanVariance(uv_errors)
    impute_moments = computeMeanVariance(impute_errors)
    times = svd_time_expectation, impute_time_expectation, uv_time_expectation
    moments = svd_moments[0], impute_moments[0], uv_moments[0]
    variances = svd_moments[1], impute_moments[1], uv_moments[1]
    medians = svd_median_expectation, soft_median_expectation, mf_median_expectation
    mins = svd_min_expectation, soft_min_expectation, mf_min_expectation
    maxs = svd_max_expecatation, soft_max_expectation, mf_max_expecatation
    print('Plotting')
    createPlots(medians, fractions, title='Median error over fraction of approximated kernel values',
                name='medians5000', xlabel='Fraction of approximated values', ylabel='Median Error')
    createPlots(mins, fractions, title='Minimum error over fraction of approximated kernel values', name='mins5000',
                xlabel='Fraction of approximated values', ylabel='Minimum error')
    createPlots(maxs, fractions, title='Maximum error over fractio of approximated kernel values', name='maxs5000',
                xlabel='Fraction of approximated values', ylabel='Maximum error')
    createPlots(times, fractions, title='Time over fraction of approximated kernel values', name='AllTime5000',
                xlabel='Fraction of approximated values', ylabel='Time in minutes')
    createPlots(moments, fractions, title='Relative mean'
                                          ' error over fraction of approximated kernel values', name="AllError5000",
                xlabel='Fraction of approximated values', ylabel='Relative RMSE')
    createPlots(variances, fractions,
                title='Variances of the relative error over fraction of approximated kernel values',
                name="AllVariance5000", xlabel='Fraction of approximated values', ylabel='Variance of relative RMSE')


if __name__ == "__main__":
    fire.Fire(startExperiment)
