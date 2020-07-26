"""
Given a pre-computed kernel and a data set, compute train/validation/test accuracy.
"""
import importlib
from copy import deepcopy
from timeit import default_timer as timer

import absl.app
import h5py
import numpy as np
import scipy
import scipy.linalg
import sklearn.metrics
import torch

from cnn_gp import DatasetFromConfig
from matrix_factorization.factorization import softImpute, iterativeSVD, plotEigenvalues

FLAGS = absl.app.flags.FLAGS


def constructSymmetricMatrix(x: np.ndarray) -> np.ndarray:
    """
    Constructs a symmetric matrix given a matrix x only with only entries in the upper triangle
    @param x: matrix only filled in upper triangle
    @return: symmetric matrix based on x
    """
    x_sym = np.empty_like(x)
    x_sym[np.triu_indices(x.shape[0], k=0)] = x[np.triu_indices(x.shape[0], k=0)]
    x_sym = x_sym + x_sym.T - np.diag(np.diag(x_sym))
    return x_sym


def computeRMSE(x: np.ndarray, y: np.ndarray) -> np.float:
    """
    @param x: original matrix which has all entries
    @param y: matrix which was filled by matrix completion algorithms
    @return: RMSE (root mean squared error) between the two matrices x and y
    """
    diff = x - y
    if np.isnan(diff).any():
        print("diff is nan")
    diff = diff ** 2
    if np.isnan(diff).any():
        print("diff^2 is nan")
    return np.sqrt(np.sum(diff) / (x.shape[0] * x.shape[1]))


def isSymmetric(x, rtol=1e-5, atol=1e-5):
    return np.allclose(x, x.T, rtol=rtol, atol=atol)


def perturbateMatrix(X):
    columns = X.shape[1]
    rows = X.shape[0]
    for row in range(rows):
        ##Sampling
        sample = np.random.randint(0, columns - 1, 3)
        X[row][sample] = np.nan
    return X


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


def print_accuracy(A, Kxvx, Y, key):
    Ypred = (Kxvx @ A).argmax(dim=1)
    acc = sklearn.metrics.accuracy_score(Y, Ypred)
    print(f"{key} accuracy: {acc * 100}%")


def load_kern(dset, i):
    A = np.empty(dset.shape[1:], dtype=np.float32)
    dset.read_direct(A, source_sel=np.s_[i, :, :])
    return torch.from_numpy(A).to(dtype=torch.float64)


def main(_):
    config = importlib.import_module(f"configs.{FLAGS.config}")
    dataset = DatasetFromConfig(FLAGS.datasets_path, config)
    computation = FLAGS.computation

    print("Reading training labels")
    _, Y = dataset.load_full(dataset.train)
    n_classes = Y.max() + 1
    Y_1hot = torch.ones((len(Y), n_classes), dtype=torch.float64).neg_()  # all -1
    Y_1hot[torch.arange(len(Y)), Y] = 1.

    with h5py.File(FLAGS.in_path, "r") as f:
        print("Loading kernel")
        Kxx = load_kern(f["Kxx"], 0)
        diag_add(Kxx, FLAGS.jitter)

        # print("Computing PCA")
        # pca_analysis(x=Kxx.cuda(), k=10)

        if computation < 1.0:
            Kxx = softImpute(Kxx)
            Kxx = torch.tensor(Kxx, dtype=torch.double)

        elif computation > 1.0:
            if torch.isnan(Kxx).any():
                Kxx_symm = constructSymmetricMatrix(Kxx.numpy())
            else:
                Kxx_symm = Kxx.numpy()
            nans = np.sum(np.isnan(Kxx_symm))
            print(f"nans in symmetric original matrix: {nans}")
            Kxx_perturbated = deepcopy(Kxx_symm)
            Kxx_perturbated = perturbateMatrix(Kxx_perturbated)
            start = timer()
            Kxx_svd = iterativeSVD(Kxx_perturbated)
            end = timer()
            diff = end - start
            print(f"svd time: {diff}")
            nans_svd = np.sum(np.isnan(Kxx_svd))
            print(f"nans in svd reconstruction: {nans_svd}")
            start = timer()
            Kxx_soft = softImpute(Kxx_perturbated)
            end = timer()
            diff = end - start
            soft_nans = np.sum(np.isnan(Kxx_soft))
            print(f"number of nans after osftimpute reconstruction: {soft_nans}")
            print(f"Soft time: {diff}")
            # start = timer()
            # #Kxx_mf = matrix_completion(Kxx_perturbated)
            # end = timer()
            # diff = end - start
            # print(f"Matrix factorization: {diff}")
            print("Kxx_svd is symmetric after applying iterative svd: " + str(isSymmetric(Kxx_svd)))
            print("Kxx_soft is symmetric after applying softImpute: " + str(isSymmetric(Kxx_soft)))
            # print("Kxx_mf is symmetric after applying MatrixFactorization: " + str(isSymmetric(Kxx_mf)))
            svd_error = computeRMSE(Kxx.numpy(), Kxx_svd)
            soft_error = computeRMSE(Kxx.numpy(), Kxx_soft)
            # mf_error = computeRMSE(Kxx.numpy(), Kxx_mf)
            print("Errors of the different methods:")
            print(f"Iterative svd error: {svd_error}")
            print(f"SoftImpute error: {soft_error}")
            # print(f"Matrix Factorization error: {mf_error}")

            del Kxx
            del Kxx_symm
            # del Kxx_mf
            del Kxx_soft
            del Kxx_perturbated
            del Kxx_svd

        else:
            Kxx_symm = constructSymmetricMatrix(Kxx.numpy())
            plotEigenvalues(Kxx_symm)

            print("Solving Kxx^{-1} Y")
            A = solve_system(Kxx_symm, Y_1hot)

            _, Yv = dataset.load_full(dataset.validation)
            Kxvx = load_kern(f["Kxvx"], 0)

            print_accuracy(A, Kxvx, Yv, "validation")
            del Kxvx
            del Yv

            _, Yt = dataset.load_full(dataset.test)
            Kxtx = load_kern(f["Kxtx"], 0)
            print_accuracy(A, Kxtx, Yt, "test")
            del Kxtx
            del Yt

            del Kxx


# @(py36) ag919@ulam:~/Programacio/cnn-gp-pytorch$ python classify_gp.py --in_path=/scratch/ag919/grams_pytorch/mnist_as_tf/00_nwork07.h5 --config=mnist_as_tf
# magma.py has some problem loading. Proceeding anyways using CPU.
# Original error: ignoring magma shit
# Reading training labels
# Loading kernel
# Solving Kxx^{-1} Y
# Running scipy solve Kxx^-1 Y routine
# train accuracy: 10.26%
# validation accuracy: 99.31%
# test accuracy: 99.11999999999999%


if __name__ == '__main__':
    f = absl.app.flags
    f.DEFINE_string("datasets_path", "/scratch/ag919/datasets/",
                    "where to save datasets")
    f.DEFINE_string("config", "mnist", "which config to load from `configs`")
    f.DEFINE_string('in_path', "/scratch/ag919/grams_pytorch/mnist/dest.h5",
                    "path of h5 file to load kernels from")
    f.DEFINE_float("jitter", 0.0, "add to the diagonal")
    f.DEFINE_float("computation", 1.0, "fraction of kxx which is computed exactly")
    absl.app.run(main)
