import os
import sys
from timeit import default_timer as timer

import fire
import h5py
import numpy as np
import torch
from fancyimpute import MatrixFactorization, SoftImpute, IterativeSVD
from torch.utils.data import Subset
from tqdm import tqdm

from cnn_gp import save_K
from matrix_factorization.nystroem import Nystroem
from plotting.createStartPlot import loadModel, loadDataset
from utils import computeRelativeRMSE, createPlots, load_kern, \
    computeMeanVariance, deleteDataset, oneHotEncoding, computePredictions, compute_accuracy, \
    readTimeandApprox, loadTargets, solve_system_fast

#### For error plot
NYSTROEM_PATH = './plotting/nystroem.h5'
ORIGINAL_PATH = './plotting/original.h5'
MF_PATH = './plotting/mf.h5'
SVD_PATH = './plotting/svd.h5'
SOFT_PATH = './plotting/soft.h5'
PERT_PATH = './plotting/Kxx_pert.h5'

FRACTIONS = np.arange(0.1, 1, 0.1)
MATRIX_SIZE = (5000, 5000)
####

## For Accuracy Plot###
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
    # precision = compute_precision(val_predictions, Y_val, key)
    # recall = compute_recall(val_predictions, Y_val, key)
    # return accuracy, precision, recall


def getFirstSamples(dataset, samples: int) -> Subset:
    """
    Returns subset of the given dataset containing only the first specified amount of points
    @param dataset: contains all points of a given dataset
    @param samples: amount of samples
    @return: subset with the first samples of the dataset
    """
    return Subset(dataset, range(samples))


def computeKernelMatrix(model, x1, x2, path, name):
    """
    Computes the kernel matrix between all given data points for a given model and returns it
    @param name: name of the dataset (so that the correct dataset is written and read from the h5py file)
    @param x2: dataset2 so that the kernel matrix is computed between all points of x1 and x2 if x2 is None,
    then only of x2 with itself (e.g Kxx matrix)
    @param x1: dataset1 for which the kernel matrix is supposed to be computed
    @param model: contains the model which is supposed to use to compute the kernel matrix
    @param path: path where the file is stored containing the kernel matrix
    @return: kernel matrix between all data points in x1 and x2
    """

    def kern(x, x2, **args):
        with torch.no_grad():
            return model(x.cuda(), x2.cuda(), **args).detach().cpu().numpy()

    """"Create file for kernel matrix and use only one worker no parallelization"""
    with h5py.File(path, "w") as f:
        kwargs = dict(worker_rank=0, n_workers=1,
                      batch_size=200, print_interval=2.)
        save_K(f, kern, name=name, X=x1, X2=x2, diag=False, **kwargs)
        f.close()
        """The loading process is included in the time measurement on purpose.
        Otherwise we could not compare it later on to methods where only a 
        part is computed and the rest is approximated, since the kernel matrix has to be loaded in any
        case
        """
        # return load_kern(f[name], 0)


def computeApproximation(Xpert, mode) -> (np.ndarray, float):
    """
    Given a matrix with missing values approximates the missing values and stops time
    @param Xpert: contains matrix where some values are nans
    @param mode: ("svd", "soft", "mf") Solver which approximates the given pertubated matrix Xpert
    @return: approximated matrix, and time for approximation
    """
    if mode == "svd":
        solver = IterativeSVD()
    elif mode == "mf":
        solver = MatrixFactorization(epochs=1000, min_improvement=0.01)
    elif mode == "soft":
        solver = SoftImpute()
    else:
        print("Mode chosen wrongly")
        sys.exit(-1)
    start = timer()
    approx = solver.fit_transform(Xpert)
    end = timer()
    diff = end - start
    minutes = diff // 60
    return approx, minutes


def compareMethodsOverError(repetitions=3):
    """
    Delete increasing fraction of original Kxx matrix and approximate it with Nystroem, SoftImpute and IterativeSVD
    Plot the errors and variances over the fractions
    @param repetitions:
    @return:
    """
    model = loadModel()
    dataset = loadDataset()
    os.system(f"python -m plotting.computeKernel computeKxxMatrix {ORIGINAL_PATH} Kxx")
    os.system(f"python -m plotting.computeKernel loadMatrixFromDiskAndMirror {ORIGINAL_PATH} Kxx")
    with h5py.File(ORIGINAL_PATH, "r") as f:
        Kxx_symm = np.empty(MATRIX_SIZE)
        f['Kxx'].read_direct(Kxx_symm)
        print('Done')
    torch.cuda.empty_cache()
    # Initializing of error and time arrays
    svd_errors = []
    svd_error = []
    svd_times = []
    svd_time = []
    nystroem_errors = []
    nystroem_error = []
    nystroem_times = []
    nystroem_time = []
    soft_errors = []
    soft_error = []
    soft_times = []
    soft_time = []
    mf_errors = []
    mf_error = []
    mf_times = []
    mf_time = []
    # Initialize
    for fraction in tqdm(FRACTIONS):
        nystroem_error.clear()
        nystroem_time.clear()
        svd_error.clear()
        svd_time.clear()
        soft_error.clear()
        soft_time.clear()
        mf_error.clear()
        mf_time.clear()
        nystroem_solver = Nystroem(n_components=int((1 - fraction) * Kxx_symm.shape[0]), k=None, model=model,
                                   dataset=dataset, path=NYSTROEM_PATH)
        for _ in tqdm(range(repetitions)):
            # Create matrix with fraction of values are randomly set to Nan
            os.system(f'python -m plotting.computeKernel computeKxxPert {ORIGINAL_PATH} {PERT_PATH} {fraction}')
            # Kxx_pert = deleteValues(Kxx_symm, fraction=fraction)
            # Write Kxx_pert to file so it can be approximated by MF:
            # with h5py.File(PERT_PATH, 'w') as f:
            #     f.create_dataset(name="Kxx_pert", shape=MATRIX_SIZE, data=Kxx_pert)
            # Try to reconstruct with different Methods
            # Iterative SVD
            print("SVD")
            os.system(f'python -m plotting.computeMatrixFactorization {PERT_PATH} {SVD_PATH} svd')
            time, approx = readTimeandApprox(SVD_PATH)
            svd_time.append(time)
            print(time)
            svd_error.append(computeRelativeRMSE(Kxx_symm, approx, fraction))
            # MF
            # print("MF")
            # os.system(f"python -m plotting.computeMatrixFactorization {PERT_PATH} {MF_PATH} mf")
            # time, approx = readTimeandApprox(MF_PATH)
            # mf_time.append(time)
            # print(time)
            # mf_error.append(computeRelativeRMSE(Kxx_symm, approx, fraction))
            # Soft Impute
            print("SOFT")
            os.system(f"python -m plotting.computeMatrixFactorization {PERT_PATH} {SOFT_PATH} soft")
            time, approx = readTimeandApprox(SOFT_PATH)
            print(time)
            soft_error.append(computeRelativeRMSE(Kxx_symm, approx, fraction))
            soft_time.append(time)
            # Nystroem
            print("NYST")
            torch.cuda.empty_cache()
            start = timer()
            approx = nystroem_solver.fit_transform()
            end = timer()
            time = (end - start) // 60
            print(time)
            torch.cuda.empty_cache()
            nystroem_time.append(time)
            nystroem_error.append(computeRelativeRMSE(Kxx_symm, approx, fraction))
            deleteDataset(NYSTROEM_PATH, None, True)
            deleteDataset(PERT_PATH, 'Kxx_pert')
        svd_errors.append(list(svd_error[:]))
        soft_errors.append(list(soft_error[:]))
        # mf_errors.append(list(mf_error[:]))
        nystroem_errors.append(list(nystroem_error[:]))
        svd_times.append(list(svd_time[:]))
        soft_times.append(list(soft_time[:]))
        # mf_times.append(list(mf_time[:]))
        nystroem_times.append(list(nystroem_time[:]))
    svd_moments = computeMeanVariance(svd_errors)
    soft_moments = computeMeanVariance(soft_errors)
    nystroem_moments = computeMeanVariance(nystroem_errors)
    # mf_moments = computeMeanVariance(mf_errors)
    svd_mean_time = computeMeanVariance(svd_times)[0]
    soft_mean_time = computeMeanVariance(soft_times)[0]
    # mf_mean_time = computeMeanVariance(mf_times)[0]
    nystroem_mean_time = computeMeanVariance(nystroem_times)[0]
    # errors = svd_moments[0], soft_moments[0], mf_moments[0], nystroem_moments[0]
    errors = svd_moments[0], soft_moments[0], nystroem_moments[0]
    # times = svd_mean_time, soft_mean_time, mf_mean_time, nystroem_mean_time
    times = svd_mean_time, soft_mean_time, nystroem_mean_time
    variances = svd_moments[1], soft_moments[1], nystroem_moments[1]
    # variances = svd_moments[1], soft_moments[1], mf_moments[1], nystroem_moments[1]
    deleteDataset(ORIGINAL_PATH)
    print('Plotting')
    createPlots(errors, FRACTIONS,
                title='Relative RMSE error over fraction of approximated 20 000 * 20 000 kernel matrix',
                name='AllError20000', xlabel='Fraction of approximated values of the kernel matrix',
                ylabel=' Relative RMSE')
    createPlots(times, FRACTIONS, title="Time in minutes to approximate fraction of 20 000*20 000  kernel matrix",
                name="AllTime20000", xlabel="Fraction of approximated values of the kernel matrix",
                ylabel="Time in minutes")
    createPlots(variances, FRACTIONS,
                title='Variance of relative RMSE over fraction of approximated 20 000*20 000 kernel matrix',
                name='AllVariances20000', xlabel='Fraction of approximated values', ylabel="Variance of relative RMSE")


""""


FILE BREAK above error, below CLASSIFICATION



"""


def compareAccuracyOverTime(repetitions=3):
    print('start')
    model = loadModel()
    dataset = loadDataset()
    os.system(f"python -m plotting.computeKernel computeKxxMatrix {ORIGINAL_PATH_ACC} Kxx")
    os.system(f"python -m plotting.computeKernel loadMatrixFromDiskAndMirror {ORIGINAL_PATH_ACC} Kxx")
    with h5py.File(ORIGINAL_PATH_ACC, "r") as f:
        Kxx_symm = np.empty(MATRIX_SIZE)
        # Kxx_symm = np.array(f.get('Kxx'))
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
    # Classification errors (Accuracy, Precision, Recall)
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
        fraction = np.sqrt(fraction)
        components = int(fraction * MATRIX_SIZE[0])
        nystroem = Nystroem(n_components=components, k=None, dataset=dataset, model=model, path=NYSTROEM_PATH_ACC)
        for rep in tqdm(range(repetitions)):
            # Compute the submatrices only once which is used for Soft, MF and SVD Iter and is fix the same
            if rep == 0:
                os.system(f"python -m plotting.computeKernel computeKxxMatrix {SUBMATRIX_PATH_ACC} Kxx_pert {fraction}")
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
            start = timer()
            approx = nystroem.fit_transform()
            end = timer()
            diff = (end - start) // 60
            nyst_time.append(diff)
            nyst_error.append(computeRelativeRMSE(Kxx_symm, approx, 1 - fraction))
            nyst_accuracy.append(evaluate(approx, Y, Kxvx, Yv, 'val'))
            deleteDataset(NYSTROEM_PATH_ACC, '', True)
        deleteDataset(SUBMATRIX_PATH_ACC, 'time')
        deleteDataset(SUBMATRIX_PATH_ACC, 'Kxx_sub')
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
    normal_accuracy = evaluate(Kxx_symm, Y, Kxvx, Yv, 'Ground truth validation')
    # Plot Reconstruction errors over fraction for different types
    print('Plotting')
    mean_errors = svd_mean_error[0], soft_mean_error[0], mf_mean_error[0], nyst_mean_error[0]
    createPlots(mean_errors, FRACTIONS,
                title='Expected relative error of approximated kernel matrix over fraction of exactly '
                      'computed '
                      'values of 5000 * 5000 matrix', name='rmse', ylabel='Relative RMSE', xlabel='Fraction of exactly'
                                                                                                  'computed '
                                                                                                  'components of the '
                                                                                                  'kernel '
                                                                                                  'matrix')
    # Plot Time over fraction for different types
    mean_times = svd_mean_time[0], soft_mean_time[0], mf_mean_time[0], nyst_mean_time[0]
    createPlots(mean_times, FRACTIONS, title='Expected computation time of approximated 5000 * 5000 kernel matrix '
                                             'over fraction '
                                             'of exactly computed elements', name='timing', ylabel='Time in minutes',
                xlabel='Fraction of exactly computed components of the kernel matrix ')
    # Plot accuracy over fraction for different types
    mean_accuracies = svd_mean_accuracy[0], soft_mean_accuracy[0], mf_mean_accuracy[0], nyst_mean_accuracy[0]
    createPlots(mean_accuracies, FRACTIONS,
                title='Expected prediction accuracy over the fraction of exactly computed values of 5000*5000 matrix',
                name='accuracy', ylabel='Accuracy', xlabel='Percentage of exactly computed components of the '
                                                           'kernel '
                                                           'matrix')


if __name__ == "__main__":
    fire.Fire()
