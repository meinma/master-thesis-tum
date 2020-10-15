import os

import h5py
import numpy as np
import scipy
import torch

from utils import load_kern, constructSymmetricMatrix


class Nystroem:
    def __init__(self, n_components, k, dataset, model, path):
        """
        Initialize Nyström solver
        @param n_components: number of m data points for which the kernel is evaluated exactly in matrix W (called M in the thesis)
        @param k: is used for the SVD only the first k most important eigenvalues are kept, others are set to zero
        @param dataset: dataset for which the nyström method is used
        @param model: model defining the kernel between two training data points
        @param path: matrices are stored there (requires to be a h5 file)
        """
        torch.cuda.empty_cache()
        self.n_components = n_components
        self.model = model
        if k is None:
            self.k = self.n_components
        else:
            self.k = k
        self.dataset = dataset
        self.path = path

    def loadMatrices(self):
        """
        @return: the matrix W and C_down which have been computed before and stored as h5py file
        """
        print("Loading")
        with h5py.File(self.path, "r") as f:
            print("Loading kernel")
            W = load_kern(f["W"], 0)
            C_down = load_kern(f["C_down"], 0, diag=True)
            f.close()
        print("Loading successful")
        return W, C_down.numpy()

    def fit_transform(self):
        """
        @return: the approximation of the original kernel matrix corresponding to
        G_k = C@W_k^+@C.T
        """
        os.system(f"python -m plotting.computeKernel computeNystroem {self.path} {self.n_components}")
        W, C_down = self.loadMatrices()
        W = constructSymmetricMatrix(W)
        # Stack both matrices row-wise
        C = np.vstack((W, C_down))
        u, sigma, vt = scipy.linalg.svd(W, full_matrices=False)
        # u, sigma, vt = np.linalg.svd(W, full_matrices=False)
        u_k = u[:, :self.k]
        sigma_k = sigma[:self.k]
        vt_k = vt[:self.k, :]
        Wk = np.linalg.pinv(u_k @ np.diag(sigma_k) @ vt_k)
        return C @ Wk @ C.T
