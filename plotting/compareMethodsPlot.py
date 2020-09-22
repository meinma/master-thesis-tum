import os
import sys
from timeit import default_timer as timer

import fire
import h5py
import numpy as np
import seaborn as sn
import torch
from fancyimpute import MatrixFactorization, SoftImpute, IterativeSVD
from torch.utils.data import Subset
from tqdm import tqdm

from cnn_gp import save_K
from matrix_factorization.nystroem import Nystroem
from plotting.createStartPlot import loadModel, loadDataset
from utils import computeRelativeRMSE, createPlots, deleteValues, \
    computeMeanVariance, deleteDataset, oneHotEncoding, solve_system, computePredictions, compute_accuracy, \
    compute_precision, compute_recall, constructSymmetricMatrix, readTimeandApprox

NYSTROEM_PATH = './plotting/nystroem.h5'
ORIGINAL_PATH = './plotting/original.h5'
MF_PATH = './plotting/mf.h5'
SVD_PATH = './plotting/svd.h5'
SOFT_PATH = './plotting/soft.h5'
PERT_PATH = './plotting/Kxx_pert.h5'

FRACTIONS = np.arange(0.1, 1, 0.1)
MATRIX_SIZE = (25000, 25000)
sn.set()


def evaluate(Kxx_approx, Y, Kxvx, Y_val, key) -> (float, float, float):
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
    A = solve_system(Kxx_approx, Y_one_hot)
    val_predictions = computePredictions(A, Kxvx)
    accuracy = compute_accuracy(val_predictions, Y_val, key)
    precision = compute_precision(val_predictions, Y_val, key)
    recall = compute_recall(val_predictions, Y_val, key)
    return accuracy, precision, recall


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
        # todo chane to f.get
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
            Kxx_pert = deleteValues(Kxx_symm, fraction=fraction)
            # Write Kxx_pert to file so it can be approximated by MF:
            with h5py.File(PERT_PATH, 'w') as f:
                f.create_dataset(name="Kxx_pert", shape=MATRIX_SIZE, data=Kxx_pert)
            # Try to reconstruct with different Methods
            # Iterative SVD
            print("SVD")
            os.system(f'python -m plotting.computeMatrixFactorization {PERT_PATH} {SVD_PATH} svd')
            time, approx = readTimeandApprox(SVD_PATH)
            svd_time.append(time)
            print(time)
            svd_error.append(computeRelativeRMSE(Kxx_symm, approx, fraction))
            print("MF")
            os.system(f"python -m plotting.computeMatrixFactorization {PERT_PATH} {MF_PATH} mf")
            time, approx = readTimeandApprox(MF_PATH)
            mf_time.append(time)
            print(time)
            mf_error.append(computeRelativeRMSE(Kxx_symm, approx, fraction))
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
        mf_errors.append(list(mf_error[:]))
        nystroem_errors.append(list(nystroem_error[:]))
        svd_times.append(list(svd_time[:]))
        soft_times.append(list(soft_time[:]))
        mf_times.append(list(mf_time[:]))
        nystroem_times.append(list(nystroem_time[:]))
    svd_moments = computeMeanVariance(svd_errors)
    soft_moments = computeMeanVariance(soft_errors)
    nystroem_moments = computeMeanVariance(nystroem_errors)
    mf_moments = computeMeanVariance(mf_errors)
    svd_mean_time = computeMeanVariance(svd_times)[0]
    soft_mean_time = computeMeanVariance(soft_times)[0]
    mf_mean_time = computeMeanVariance(mf_times)[0]
    nystroem_mean_time = computeMeanVariance(nystroem_times)[0]
    errors = svd_moments[0], soft_moments[0], mf_moments[0], nystroem_moments[0]
    times = svd_mean_time, soft_mean_time, mf_mean_time, nystroem_mean_time
    variances = svd_moments[1], soft_moments[1], mf_moments[1], nystroem_moments[1]
    deleteDataset(ORIGINAL_PATH)
    print('Plotting')
    createPlots(errors, FRACTIONS,
                title='Relative RMSE error over fraction of approximated 25 000 * 25 000 kernel matrix',
                name='AllError25000', xlabel='Fraction of approximated values of the kernel matrix',
                ylabel=' Relative RMSE')
    createPlots(times, FRACTIONS, title="Time in minutes to approximate fraction of 25 000*25 000  kernel matrix",
                name="AllTime25000", xlabel="Fraction of approximated values of the kernel matrix",
                ylabel="Time in minutes")
    createPlots(variances, FRACTIONS,
                title='Variance of relative RMSE over fraction of approximated 25 000*25 000 kernel matrix',
                name='AllVariances25000', xlabel='Fraction of approximated values', ylabel="Variance of relative RMSE")


""""


FILE BREAK above error, below CLASSIFICATION



"""


# todo might be a good idea to have different paths for nystroem and not nystroem
def measureTime(model, dataset, fraction, mode, path) -> (float, np.ndarray):
    print(f'Measure time for fraction {fraction}')
    inverse_fraction = 1 - fraction
    length = len(dataset)
    components = int(length * inverse_fraction)
    subset = getFirstSamples(dataset, components)
    nystroem_path = path + '/nystroem'
    if mode == 'nyst':
        print('Nystroem')
        start = timer()
        nystroem = Nystroem(n_components=components, k=None, dataset=dataset, model=model,
                            path=nystroem_path)
        erg = nystroem.fit_transform()
        end = timer()
        deleteDataset(path, True)
        diff = end - start
    else:
        start = timer()
        subMatrix = computeKernelMatrix(model, x1=subset, x2=None, path=path, name='Kxx')
        subMatrix = constructSymmetricMatrix(subMatrix)
        # subMatrix = computeKernelMatrixParallel(model=model, x1=subset, x2=None, path=path, name='Kxx', diag=False)
        finalMatrix = np.empty((length, length))
        finalMatrix[0:subMatrix.shape[0], 0:subMatrix.shape[1]] = subMatrix
        if mode == 'svd':
            erg = iterativeSVD(finalMatrix)
        else:
            erg = softImpute(finalMatrix)
        end = timer()
        deleteDataset(path)
        diff = end - start
    diff = diff // 60  # return time in minutes
    print(diff)
    return diff, erg


# def compareMethodsOverTime(repetitions=5):  # todo adjust the fractions (0.5 corresponds to 0.25 actually)
#     print('start')
#     path = './plotting/approximations'
#     fractions = np.arange(0.1, 1, 0.1)
#     model = loadModel()
#     full_dataset = loadDataset()
#     train_dataset = getFirstSamples(full_dataset, 25000)
#     validation_set = Subset(full_dataset, list(range(25000, 27500)))
#     train_labels = loadTargets(train_dataset)
#     validation_labels = loadTargets(validation_set)
#     # Compute the kernel matrix exactly
#     # Kxx_symm = computeKernelMatrixParallel(model, x1=train_dataset, x2=None, path='./plotting/orig', name='Kxx')
#     Kxx_orig = computeKernelMatrix(model, x1=train_dataset, x2=None, path=ORIGINAL_PATH, name='Kxx').numpy()
#     Kxx_symm = constructSymmetricMatrix(Kxx_orig)
#     # Kxvx = computeKernelMatrixParallel(model, x1=validation_set, x2=train_dataset,
#     #                                    path=ORIGINAL_PATH, name='Kxvx', diag=True)
#     # Initialization of all arrays containing the times, errors, evaluation measurements and so on
#     # return
#     svd_times = []
#     nyst_times = []
#     soft_times = []
#     svd_time = []
#     nyst_time = []
#     soft_time = []
#     svd_error = []
#     svd_errors = []
#     soft_error = []
#     soft_errors = []
#     nyst_error = []
#     nyst_errors = []
#     # Classification errors (Accuracy, Precision, Recall)
#     svd_accuracies = []
#     soft_accuracies = []
#     nyst_accuracies = []
#     svd_accuracy = []
#     soft_accuracy = []
#     nyst_accuracy = []
#     svd_precisions = []
#     soft_precisions = []
#     nyst_precisions = []
#     svd_precision = []
#     soft_precision = []
#     nyst_precision = []
#     svd_recalls = []
#     soft_recalls = []
#     nyst_recalls = []
#     svd_recall = []
#     soft_recall = []
#     nyst_recall = []
#     for fraction in tqdm(fractions):
#         svd_time.clear()
#         nyst_time.clear()
#         soft_time.clear()
#         svd_error.clear()
#         soft_error.clear()
#         nyst_error.clear()
#         svd_accuracy.clear()
#         soft_accuracy.clear()
#         nyst_accuracy.clear()
#         svd_precision.clear()
#         soft_precision.clear()
#         nyst_precision.clear()
#         svd_recall.clear()
#         soft_recall.clear()
#         nyst_recall.clear()
#         for _ in tqdm(range(repetitions)):
#             time, approximation = measureTime(model, train_dataset, fraction, 'svd', path)
#             svd_time.append(time)
#             svd_error.append(computeRMSE(Kxx_symm, approximation))
#             acc, prec, recall = evaluate(Kxx_approx=approximation, Y=train_labels, Kxvx=Kxvx,
#                                          Y_val=validation_labels, key='Validation')
#             svd_accuracy.append(acc)
#             svd_precision.append(prec)
#             svd_recall.append(recall)
#             time, approximation = measureTime(model, train_dataset, fraction, 'soft', path)
#             soft_time.append(time)
#             soft_error.append(computeRMSE(Kxx_symm, approximation))
#             acc, prec, recall = evaluate(approximation, train_labels, Kxvx, validation_labels, key='validation')
#             soft_accuracy.append(acc)
#             soft_precision.append(prec)
#             soft_recall.append(recall)
#             time, approximation = measureTime(model, train_dataset, fraction, 'nyst', path)
#             nyst_time.append(time)
#             nyst_error.append(computeRMSE(Kxx_symm, approximation))
#             acc, prec, recall = evaluate(approximation, train_labels, Kxvx, validation_labels, key='validation')
#             nyst_accuracy.append(acc)
#             nyst_precision.append(prec)
#             nyst_recall.append(recall)
#         svd_times.append(svd_time)
#         soft_times.append(soft_time)
#         nyst_times.append(nyst_time)
#         svd_errors.append(svd_error)
#         soft_errors.append(soft_error)
#         nyst_errors.append(nyst_error)
#         svd_accuracies.append(svd_accuracy)
#         soft_accuracies.append(soft_accuracy)
#         nyst_accuracies.append(nyst_accuracy)
#         svd_precisions.append(svd_precision)
#         soft_precisions.append(soft_precision)
#         nyst_precisions.append(nyst_precision)
#         svd_recalls.append(svd_recall)
#         soft_recalls.append(soft_recall)
#         nyst_recalls.append(nyst_recall)
#     svd_mean_time = computeMeanVariance(svd_times)
#     soft_mean_time = computeMeanVariance(soft_times)
#     nyst_mean_time = computeMeanVariance(nyst_times)
#     svd_mean_error = computeMeanVariance(svd_errors)
#     soft_mean_error = computeMeanVariance(soft_errors)
#     nyst_mean_error = computeMeanVariance(nyst_errors)
#     svd_mean_accuracy = computeMeanVariance(svd_accuracies)
#     soft_mean_accuracy = computeMeanVariance(soft_accuracies)
#     nyst_mean_accuracy = computeMeanVariance(nyst_accuracies)
#     svd_mean_precision = computeMeanVariance(svd_precisions)
#     soft_mean_precision = computeMeanVariance(soft_precisions)
#     nyst_mean_precision = computeMeanVariance(nyst_precisions)
#     svd_mean_recall = computeMeanVariance(svd_recalls)
#     soft_mean_recall = computeMeanVariance(soft_recalls)
#     nyst_mean_recall = computeMeanVariance(nyst_recalls)
#     # Plot Reconstruction errors over fraction for different types
#     print('Plotting')
#     mean_errors = svd_mean_error[0], soft_mean_error[0], nyst_mean_error[0]
#     createPlots(mean_errors, fractions * 100, title='Expected error of approximated kernel matrix and exactly computed '
#                                                     'one', name='rmse', ylabel='RMSE', xlabel='Percentage of missing '
#                                                                                               'values of the kernel '
#                                                                                               'matrix')
#     # Plot Time over fraction for different types
#     mean_times = svd_mean_time[0], soft_mean_time[0], nyst_mean_time[0]
#     createPlots(mean_times, fractions * 100, title='Expected computation time of kernel matrix over percentage of '
#                                                    'missing values', name='timing', ylabel='Time in minutes',
#                 xlabel='Percentage of missing values of the kernel matrix ')
#     # Plot accuracy over fraction for different types
#     mean_accuracies = svd_mean_accuracy[0], soft_mean_accuracy[0], nyst_mean_accuracy[0]
#     createPlots(mean_accuracies, fractions * 100,
#                 title='Expected prediction accuracy over the percentage of missing values',
#                 name='accuracy', ylabel='Accuracy', xlabel='Percentage of missing values of the kernel matrix')
#     mean_precisions = svd_mean_precision[0], soft_mean_precision[0], nyst_mean_precision[0]
#     createPlots(mean_precisions, fractions * 100, title='Expected prediction precision over percentage of missing '
#                                                         'values of the kernel matrix', name='precision',
#                 ylabel='Precision', xlabel='Percentage of approximated values of kernel matrix')
#     mean_recalls = svd_mean_recall[0], soft_mean_recall[0], nyst_mean_recall[0]
#     createPlots(mean_recalls, fractions * 100,
#                 title='Expected prediction recall over percentage of approximated values '
#                       'for kernel matrix',
#                 name='recall', ylabel='Recall', xlabel='Percentage of approximated values of kernel matrix')


if __name__ == "__main__":
    fire.Fire()
    # compareMethodsOverTime(path)
    # compareMethodsOverError(path)
