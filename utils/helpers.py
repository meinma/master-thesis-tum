import copy
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sn
import torch
from scipy.sparse.linalg import lsmr
from sklearn import metrics
from torch.utils.data import DataLoader

from configs import mnist_paper_convnet_gp
from utils.Error import Error

sn.set()

__all__ = ('createPlots', 'plotEigenvalues', 'oneHotEncoding', 'computeRMSE', 'compute_accuracy',
           'computeMeanVariance', 'constructSymmetricIfNotSymmetric', 'computePredictions',
           'constructSymmetricMatrix', 'isSymmetric', 'print_accuracy', 'load_kern', 'loadTargets', 'deleteDataset',
           'solve_system_old', 'deleteValues', 'diag_add', 'generateSquareRandomMatrix', 'solve_system_fast',
           'computeRelativeRMSE', 'readTimeandApprox', 'computeErrors', 'loadOriginalModel', 'loadNormalizedModel')


def createPlots(moments, fractions, title, name, xlabel, ylabel):
    """
    Creates plots for the given moment data on y and fractions data on x with name as title
    @param title: sets the title
    @param moments: contains expected values or variances of the errors
    of the matrix approximation methods on the y-axis
    @param fractions: contains the values for the x-axis
    @param name: specifies name for the plot
    @return: None
    """
    plt.figure()
    plt.rcParams.update({'axes.titlesize': 'x-small'})
    plt.title(f"{title}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(fractions, moments[0], label='Iterative svd')
    plt.plot(fractions, moments[1], label='Soft impute')
    plt.plot(fractions, moments[2], label='Matrix Factorization')
    if len(moments) > 3:
        plt.plot(fractions, moments[3], label='Nystroem Approximation')
    plt.legend()
    plt.savefig(f'./plots/{name}.svg')


def plotEigenvalues(x: np.ndarray):
    """
    @param x: contains the matrix for which the eigenvalues are computed
    @return: plots the eigenvalues of the given matrix x
    """
    print("Plotting eigenvalues")
    eigenvalues, _ = np.linalg.eigh(x)
    eigenvalues = eigenvalues[::-1]
    max_val = np.max(eigenvalues)
    smaller_vals = eigenvalues / max_val
    print(eigenvalues)
    plt.figure()
    plt.plot(eigenvalues)
    plt.xlabel("Number of eigenvalue")
    plt.ylabel("Eigenvalue")
    plt.title("Plot of the eigenvalues of the K_xx matrix")
    plt.show()
    plt.savefig('./plots/eigenvalues.svg')
    plt.close()
    plt.figure()
    plt.plot(np.log(eigenvalues))
    plt.xlabel('Number of the eigenvalue')
    plt.ylabel('Log space of eigenvalue')
    plt.title('Plot of the eigenvalues of the K_xx matrix in the log space')
    plt.show()
    plt.savefig('./plots/log_eigenvalues.svg')
    plt.close()
    plt.figure()
    plt.title("Plot of the normalized eigenvalues")
    plt.xlabel("Number of eigenvalue")
    plt.ylabel("Magnitude of normalized eigenvalues")
    plt.plot(np.log(smaller_vals))
    plt.show()
    plt.savefig('./plots/normalized_eigenvalues.svg')
    plt.close()


def oneHotEncoding(Y):
    """
    Creates one hot encoding from given target vector
    @param Y: target vector
    @return: one hot encoding corresponding to Y
    """
    n_classes = Y.max() + 1
    Y_1hot = torch.ones((len(Y), n_classes), dtype=torch.float64).neg_()  # all -1
    Y_1hot[torch.arange(len(Y)), Y] = 1.
    return Y_1hot


def constructSymmetricIfNotSymmetric(x: np.ndarray) -> np.ndarray:
    """
    Checks if the given matrix x is symmetric if not, constructs a symmetric matrix from the upper triangle
    @param x: matrix which is checked and created symmetric
    @return: symmetric matrix of x if x is unsymmetric
    """
    if isSymmetric(x):
        return x
    else:
        return constructSymmetricMatrix(x)


def constructSymmetricMatrix(x: torch.float64) -> np.ndarray:
    """
    Constructs a symmetric matrix given a matrix x only with only entries in the upper triangle
    @param x: matrix only filled in upper triangle
    @return: symmetric matrix based on x
    """
    assert x.dtype == torch.float64
    x_sym = np.empty_like(x)
    x_sym[np.triu_indices(x.shape[0], k=0)] = x[np.triu_indices(x.shape[0], k=0)]
    x_sym = x_sym + x_sym.T - np.diag(np.diag(x_sym))
    return x_sym


def isSymmetric(x, rtol=1e-5, atol=1e-5):
    return np.allclose(x, x.T, rtol=rtol, atol=atol)


def solve_system_fast(Kxx: np.ndarray, Y: np.ndarray) -> torch.float64:
    """
    Inverts the Kxx matrix
    @param Kxx: kernel matrix
    @param Y: one hot encoded target matrix
    @return: The inverse of Kxx as torch
    """
    solution = [lsmr(Kxx, Y[:, k])[0] for k in range(Y.shape[1])]
    return torch.from_numpy(np.column_stack(solution))


def solve_system(Kxx: np.ndarray, Y) -> torch.float64:
    print("Running scipy solve Kxx^-1 Y routine")
    assert Y.dtype == torch.float64, """
    It is important that `Kxx` and `Y` are `float64`s for the inversion,
    even if they were `float32` when being calculated. This makes the
    inversion much less likely to complain about the matrix being singular.
    """
    A, _, _, _ = scipy.linalg.lstsq(
        Kxx, Y.numpy())
    return torch.from_numpy(A)


def solve_system_old(Kxx, Y):
    print("Running scipy solve Kxx^-1 Y routine")
    assert Kxx.dtype == torch.float64 and Y.dtype == torch.float64, """
    It is important that `Kxx` and `Y` are `float64`s for the inversion,
    even if they were `float32` when being calculated. This makes the
    inversion much less likely to complain about the matrix being singular.
    """
    A = scipy.linalg.solve(
        Kxx.numpy(), Y.numpy(), overwrite_a=True, overwrite_b=False,
        check_finite=False, assume_a='pos', lower=False)
    return torch.from_numpy(A)


def diag_add(K, diag):
    if isinstance(K, torch.Tensor):
        K.view(K.numel())[::K.shape[-1] + 1] += diag
    elif isinstance(K, np.ndarray):
        K.flat[::K.shape[-1] + 1] += diag
    else:
        raise TypeError("What do I do with a `{}`, K={}?".format(type(K), K))


def computePredictions(A, Kxvx):
    """
    Computes Kxvx Kxx^-1 Y to obtain the predictions for the points xv
    Computes the predictions of Gaussian Processes Given A = Kxx^(-1) * Y
    and the kernel of the data points for the predictions
    @param A: the result of Kxx^-1 Y
    @param Kxvx: the kernel matrix of the unobserved data points and the observed ones
    @return: predictions for the unobserved data points
    """
    return (Kxvx @ A).argmax(dim=1)


def compute_recall(Y_pred, Y, key):
    """
    Computes the recall given the the predictions and the ground truth labels
    @param Y_pred: predictions of data points
    @param Y: ground truth labels of data points
    @return: recall
    """
    recall = metrics.recall_score(Y, Y_pred, average='micro')
    print(f"{key} recall: {recall * 100}%")
    return recall


def compute_precision(Y_pred, Y, key):
    """
    Computes the precision given the predictions and the ground truth labels
    @param Y_pred: predictions of data points
    @param Y: ground truth labels of data points
    @param key: specifies for which dataset the predictions were generated
    @return: precision
    """
    precision = metrics.precision_score(Y, Y_pred, average='micro')
    print(f"{key} precision: {precision * 100}%")
    return precision


def compute_accuracy(Y_pred, Y, key):
    """
    returns the accuracy, given the predictions and the ground truth labels
    @param Y_pred: predictions
    @param Y: ground truth labels
    @param key: 'val' or 'test' to print
    @return: accuracy score
    """
    accuracy = metrics.accuracy_score(Y, Y_pred)
    print(f"{key} accuracy: {accuracy * 100}%")
    return accuracy


def print_accuracy(A, Kxvx, Y, key):
    """
    Computing the predictions and printing the accuracy
    @param A: The inverse of Kxx
    @param Kxvx: Kernel matrix between the training and validation/test data
    @param Y: ground truth labels
    @param key: ''validation' or 'test' used for the printing
    @return:
    """
    Ypred = (Kxvx @ A).argmax(dim=1)
    acc = metrics.accuracy_score(Y, Ypred)
    print(f"{key} accuracy: {acc * 100}%")


def load_kern(dset, i, diag=False):
    A = np.empty(dset.shape[1:], dtype=np.float32)
    if diag:
        dset.read_direct(A, source_sel=np.s_[i, :])
    else:
        dset.read_direct(A, source_sel=np.s_[i, :, :])
    return torch.from_numpy(A).to(dtype=torch.float64)


def computeRMSE(x: np.ndarray, y: np.ndarray, fraction) -> np.float:
    """
    @param fraction: fraction of predicted  values
    @param x: original matrix which has all entries
    @param y: matrix which was filled by matrix completion algorithms
    @return: RMSE (root mean squared error) between the two matrices x and y
    """
    N = int(fraction * y.shape[0] * y.shape[1])
    diff = x - y
    diff = diff ** 2
    return np.sqrt(np.sum(diff) / N)


def computeRelativeRMSE(orig, approx, fraction) -> np.float:
    """
    Computes the relative RMSE
    @param orig: contains the true matrix which is approximated
    @param approx: approximation of x
    @param fraction: fraction of values which is approximated of x
    @return: relative RMSE
    """
    weighting = 1 / (np.max(orig) - np.min(orig))
    return weighting * computeRMSE(orig, approx, fraction)


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


def deleteDataset(path, name='Kxx', nyst=False):
    """
        Deletes the h5py dataset given by the path
        @param name: specifies dataset within file which is supposed to be deleted
        @param path: defines the path to the dataset which is supposed to be deleted
        @return:
        """
    with h5py.File(path, 'a') as f:
        # del f[name]
        if nyst:
            del f['W']
            del f['C_down']
        else:
            if name == 'Kxx':
                del f['Kxx']
            else:
                del f[name]


def readTimeandApprox(path):
    """
    Reads time and approximation from given file
    @param path: Path of the file
    @return: numpy array from h5py file
    """
    with h5py.File(path, 'r') as f:
        time = np.array(f.get('time'))
        approx = np.array(f.get('approx'))
    deleteDataset(path, 'time')
    deleteDataset(path, 'approx')
    return time, approx


def loadTargets(dataset):
    """
    Return labels for a given dataset
    @param dataset: containing data and its corresponding labels
    @return: labels
    """
    _, Y = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    return Y


def computeErrors(x: np.ndarray, y: np.ndarray) -> Error:
    """
    Given two matrices computes the max, min and median error and returns an Error object containing them all
    @param x: Original matrix
    @param y: Approximated matrix
    @return: Error object containing min, max and median error
    """
    diff = np.abs(x - y)
    min_error = np.min(diff)
    max_error = np.max(diff)
    median_error = np.median(diff)
    return Error(min_error, max_error, median_error)


def loadOriginalModel(config=mnist_paper_convnet_gp):
    """
    Loads the original model from the given config
    @return: model on GPU
    """
    return config.initial_model.cuda()


def loadNormalizedModel():
    """
    Returns the model that includes the normalization layer before every ReLU
    @param config: by default the mnist config, can be changed to other configs
    @return:
    """
    return mnist_paper_convnet_gp.normalized_model.cuda()
