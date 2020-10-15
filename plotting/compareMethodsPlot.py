import os
from timeit import default_timer as timer

import fire
import h5py
import numpy as np
from tqdm import tqdm

from matrix_factorization.nystroem import Nystroem
from plotting.createStartPlot import loadDataset
from utils import computeRelativeRMSE, createPlots, load_kern, \
    computeMeanVariance, deleteDataset, oneHotEncoding, computePredictions, compute_accuracy, \
    readTimeandApprox, loadTargets, solve_system_fast, loadNormalizedModel

FRACTIONS = np.arange(0.1, 1.0, 0.1)
MATRIX_SIZE = (5000, 5000)
####


NYSTROEM_PATH_ACC = './plotting/accuracy_files/nystroem.h5'
ORIGINAL_PATH_ACC = './plotting/accuracy_files/original.h5'  # for Kxx and Kxvx
MF_PATH_ACC = './plotting/accuracy_files/mf.h5'
SVD_PATH_ACC = './plotting/accuracy_files/svd.h5'
SOFT_PATH_ACC = './plotting/accuracy_files/soft.h5'
SUBMATRIX_PATH_ACC = './plotting/accuracy_files/Kxx_pert.h5'


def evaluate(Kxx_approx, Y, Kxvx, Y_val, key) -> float:
    """
    Returns accuracy, precision and recall for an approximated kernel matrix Kxx
    @param key: specify what you compute ('validation' or 'test')
    @param Y: labels of training data
    @param Kxx_approx: Approximation of kernel matrix Kxx
    @param Kxvx: Exactly computed kernel matrix between data and validation points
    @param Y_val: validation labels (not one hot encoded)
    @return: accuracy, precision, recall
    """
    Y_one_hot = oneHotEncoding(Y)
    A = solve_system_fast(Kxx_approx, Y_one_hot.numpy())
    val_predictions = computePredictions(A, Kxvx)
    accuracy = compute_accuracy(val_predictions, Y_val, key)
    return accuracy


def compareAccuracyOverTime(repetitions=3):
    print('start')
    model = loadNormalizedModel()
    dataset = loadDataset()
    os.system(f"python -m plotting.computeKernel computeKxxMatrix {ORIGINAL_PATH_ACC} Kxx")
    os.system(f"python -m plotting.computeKernel loadMatrixFromDiskAndMirror {ORIGINAL_PATH_ACC} Kxx")
    with h5py.File(ORIGINAL_PATH_ACC, "r") as f:
        Kxx_symm = np.empty(MATRIX_SIZE)
        f['Kxx'].read_direct(Kxx_symm)
        print('Done')
        f.close()
    os.system(f"python -m plotting.computeKernel computeValidationKernel {ORIGINAL_PATH_ACC} Kxvx")
    with h5py.File(ORIGINAL_PATH_ACC, 'r') as f:
        Kxvx = load_kern(f['Kxvx'], 0)
    Y = loadTargets(dataset)
    Yv = loadTargets(loadDataset(mode='val'))
    svd_times = []
    nyst_times = []
    soft_times = []
    mf_times = []
    svd_time = []
    nyst_time = []
    soft_time = []
    mf_time = []
    svd_error = []
    svd_errors = []
    soft_error = []
    soft_errors = []
    nyst_error = []
    nyst_errors = []
    mf_error = []
    mf_errors = []
    svd_accuracies = []
    soft_accuracies = []
    nyst_accuracies = []
    mf_accuracies = []
    svd_accuracy = []
    soft_accuracy = []
    nyst_accuracy = []
    mf_accurracy = []
    for fraction in tqdm(FRACTIONS):
        svd_time.clear()
        nyst_time.clear()
        soft_time.clear()
        mf_time.clear()
        svd_error.clear()
        soft_error.clear()
        nyst_error.clear()
        mf_error.clear()
        svd_accuracy.clear()
        soft_accuracy.clear()
        nyst_accuracy.clear()
        mf_accurracy.clear()
        # Compute the proper fraction because taking half of the components does only cover one fourth of the matrix
        # Therefore the sqare root is taken here in order to obtain the fraction of the matrix
        dataset_fraction = np.sqrt(fraction)
        components = int(fraction * MATRIX_SIZE[0])
        nystroem = Nystroem(n_components=components, k=None, dataset=dataset, model=model, path=NYSTROEM_PATH_ACC)
        for rep in tqdm(range(repetitions)):
            # Compute the submatrices only once which is used for Soft, MF and SVD Iter and is fix the same
            if rep == 0:
                os.system(
                    f"python -m plotting.computeKernel computeKxxMatrix {SUBMATRIX_PATH_ACC} Kxx_pert {dataset_fraction}")
                with h5py.File(SUBMATRIX_PATH_ACC, 'r') as f:
                    basic_timing = np.array(f.get('time'))
            ## SVD
            os.system(f"python -m plotting.computeMatrixFactorization {SUBMATRIX_PATH_ACC} {SVD_PATH_ACC} svd")
            time, approx = readTimeandApprox(SVD_PATH_ACC)
            timing = basic_timing + time
            svd_time.append(timing)
            svd_error.append(computeRelativeRMSE(Kxx_symm, approx, fraction=1 - fraction))
            svd_accuracy.append(evaluate(Kxx_approx=approx, Y=Y, Kxvx=Kxvx, Y_val=Yv, key='validation'))
            ## SOFT
            os.system(f"python -m plotting.computeMatrixFactorization {SUBMATRIX_PATH_ACC} {SOFT_PATH_ACC} soft")
            time, approx = readTimeandApprox(SOFT_PATH_ACC)
            soft_time.append(time + basic_timing)
            soft_error.append(computeRelativeRMSE(Kxx_symm, approx, fraction=1 - fraction))
            soft_accuracy.append(evaluate(Kxx_approx=approx, Y=Y, Kxvx=Kxvx, Y_val=Yv, key='validation'))
            ##Mf
            os.system(f"python -m plotting.computeMatrixFactorization {SUBMATRIX_PATH_ACC} {MF_PATH_ACC}  mf")
            time, approx = readTimeandApprox(MF_PATH_ACC)
            mf_time.append(time + basic_timing)
            mf_error.append(computeRelativeRMSE(Kxx_symm, approx, fraction=1 - fraction))
            mf_accurracy.append(evaluate(approx, Y, Kxvx, Yv, 'val'))
            ## NYST
            print(rep)
            print(fraction)
            start = timer()
            approx = nystroem.fit_transform()
            end = timer()
            diff = (end - start) / 60
            nyst_time.append(diff)
            nyst_error.append(computeRelativeRMSE(Kxx_symm, approx, 1 - fraction))
            nyst_accuracy.append(evaluate(approx, Y, Kxvx, Yv, 'val'))
            deleteDataset(NYSTROEM_PATH_ACC, '', True)
        deleteDataset(SUBMATRIX_PATH_ACC, 'time')
        deleteDataset(SUBMATRIX_PATH_ACC, 'Kxx_pert')
        svd_times.append(list(svd_time[:]))
        soft_times.append(list(soft_time[:]))
        nyst_times.append(list(nyst_time[:]))
        mf_times.append(list(mf_time[:]))
        svd_errors.append(list(svd_error[:]))
        soft_errors.append(list(soft_error[:]))
        nyst_errors.append(list(nyst_error[:]))
        mf_errors.append(list(mf_error[:]))
        svd_accuracies.append(list(svd_accuracy[:]))
        soft_accuracies.append(list(soft_accuracy[:]))
        nyst_accuracies.append(list(nyst_accuracy[:]))
        mf_accuracies.append(list(mf_accurracy[:]))
    svd_mean_time = computeMeanVariance(svd_times)
    soft_mean_time = computeMeanVariance(soft_times)
    nyst_mean_time = computeMeanVariance(nyst_times)
    mf_mean_time = computeMeanVariance(mf_times)
    svd_mean_error = computeMeanVariance(svd_errors)
    soft_mean_error = computeMeanVariance(soft_errors)
    nyst_mean_error = computeMeanVariance(nyst_errors)
    mf_mean_error = computeMeanVariance(mf_errors)
    svd_mean_accuracy = computeMeanVariance(svd_accuracies)
    soft_mean_accuracy = computeMeanVariance(soft_accuracies)
    nyst_mean_accuracy = computeMeanVariance(nyst_accuracies)
    mf_mean_accuracy = computeMeanVariance(mf_accuracies)
    evaluate(Kxx_symm, Y, Kxvx, Yv, 'Ground truth validation')
    # Plot Reconstruction errors over fraction for different types
    print('Plotting')
    mean_errors = np.log(svd_mean_error[0]), np.log(soft_mean_error[0]), np.log(mf_mean_error[0]), np.log(
        nyst_mean_error[0])
    createPlots(mean_errors, FRACTIONS,
                title='Expected relative error of approximated and exactly computed 5000 * 5000 kernel matrix'
                , name='rmse_5', ylabel='Relative RMSE',
                xlabel='Fraction of exactly computed values of the kernel matrix')
    # Plot Time over fraction for different types
    mean_times = svd_mean_time[0], soft_mean_time[0], mf_mean_time[0], nyst_mean_time[0]
    createPlots(mean_times, FRACTIONS,
                title='Expected computation time for approximating 5000 * 5000 kernel matrix exactly computed values',
                name='timing_5', ylabel='Time in minutes',
                xlabel='Fraction of exactly computed values of the kernel matrix ')
    # Plot accuracy over fraction for different types
    mean_accuracies = svd_mean_accuracy[0], soft_mean_accuracy[0], mf_mean_accuracy[0], nyst_mean_accuracy[0]
    createPlots(mean_accuracies, FRACTIONS,
                title='Expected prediction accuracy over the fraction of exactly computed values of 5000 * 5000 matrix',
                name='accuracy_5', ylabel='Accuracy', xlabel='Fraction of exactly computed values of the '
                                                             'kernel '
                                                             'matrix')


if __name__ == "__main__":
    fire.Fire()
