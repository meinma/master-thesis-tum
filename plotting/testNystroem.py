from timeit import default_timer as timer

import h5py
import numpy as np
import torch
from torch.utils.data import Subset

from cnn_gp import save_K
from plotting.createStartPlot import loadDataset, loadModel
from utils import load_kern, constructSymmetricMatrix, computeRelativeRMSE


def computeKxx(model, dataset, path):
    def kern(x, x2, **args):
        with torch.no_grad():
            return model(x.cuda(), x2.cuda(), **args).detach().cpu().numpy()

    kwargs = dict(worker_rank=0, n_workers=1,
                  batch_size=200, print_interval=2.)
    with h5py.File(path, "w") as f:
        save_K(f, kern, "Kxx", X=dataset, X2=None, diag=False, **kwargs)


def computeKernelMatrices(model, dataset, subset, out_path):
    """ Computes the C matrix containing the kernel between the sampled points and the all data points
    in the training set. The matrix is stored in a h5py file.
    Computes the diagonal additionally to obtain more accurate results
    @return:
    """

    def kern(x, x2, **args):
        with torch.no_grad():
            return model(x.cuda(), x2.cuda(), **args).detach().cpu().numpy()

    with h5py.File(out_path, "w") as f:
        kwargs = dict(worker_rank=0, n_workers=1,
                      batch_size=200, print_interval=2.)
        # rectangular matrix consisting of W and S
        save_K(f, kern, name="C", X=dataset, X2=subset, diag=False, **kwargs)
        #  Compute the diagonal for C for better accuracy
        save_K(f, kern, name="Cd", X=dataset, X2=None, diag=True, **kwargs)


def Variant2(model, dataset, subset, path):
    def kern(x, x2, **args):
        with torch.no_grad():
            return model(x.cuda(), x2.cuda(), **args).detach().cpu().numpy()

    dataset_low = Subset(dataset, range(len(subset), len(dataset)))

    with h5py.File(path, 'w') as f:
        kwargs = dict(worker_rank=0, n_workers=1,
                      batch_size=200, print_interval=2.)
        # Compute squared part first
        save_K(f, kern, 'W', X=subset, X2=None, diag=False, **kwargs)
        # Lower part
        save_K(f, kern, "C_down", X=dataset_low, X2=subset, diag=False, **kwargs)
        # Lower part diag
        # save_K(f, kern, "C_down_diag", X=subset, X2=, diag=True, **kwargs)


if __name__ == "__main__":
    total = 5000
    components = 2000
    matrix2 = np.empty((total, components))
    model = loadModel()
    dataset = loadDataset()
    dataset = Subset(dataset, range(total))
    subset = Subset(dataset, range(components))
    path1 = "./scratch/ny_test.h5"
    start = timer()
    computeKernelMatrices(model, dataset, subset, path1)
    with h5py.File(path1, 'r') as f:
        C = load_kern(f["C"], 0).numpy()
        C_diag = load_kern(f["Cd"], 0, True).numpy()
    np.fill_diagonal(C, C_diag[:components])
    matrix = C
    end = timer()
    diff = (end - start) // 60
    path2 = "./scratch/ny_test2.h5"
    start = timer()
    Variant2(model, dataset, subset, path2)
    with h5py.File(path2, 'r') as f:
        W = load_kern(f["W"], 0)
        W = constructSymmetricMatrix(W)
        C_down = load_kern(f["C_down"], 0).numpy()
    matrix2 = np.vstack((W, C_down))
    # matrix2[:components, :components] = W[:, :]
    # matrix2[components:total, :components] = C_down[:, :]
    end = timer()
    path_orig = "./scratch/test_orig.h5"
    computeKxx(model, dataset, path_orig)
    with h5py.File(path_orig, "r") as f:
        Kxx = load_kern(f["Kxx"], 0)
        Kxx = constructSymmetricMatrix(Kxx)
    Kxx_comp = Kxx[:, :components]
    diff2 = (end - start) // 60
    comp1 = computeRelativeRMSE(Kxx_comp, matrix, 1)
    comp2 = computeRelativeRMSE(Kxx_comp, matrix2, 1)
    print(f"First variant: {diff}")
    print(f"Second variant: {diff2}")
    print(f"Comparison between orig and matrix 1. Error: {comp1}")
    print(f"Comparison between orig and matrix 2. Error: {comp2}")

# Zweite Variante schneller und exakt
