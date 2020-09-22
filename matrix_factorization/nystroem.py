import os

import h5py
import numpy as np
import torch

from utils import load_kern


class Nystroem:
    def __init__(self, n_components, k, dataset, model, path):
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
        @return: the matrix C and C_diag which have been computed before and stored as h5py file
        """
        print("Loading")
        with h5py.File(self.path, "r") as f:
            print("Loading kernel")
            C = load_kern(f["C"], 0)
            C_diag = load_kern(f["Cd"], 0, diag=True)
            f.close()
        print("Loading successful")
        return C.numpy(), C_diag.numpy()

    def fit_transform(self):
        """
        @return: the approximation of the original kernel matrix corresponding to
        G_k = C@W_k^+@C.T
        """
        os.system(f"python -m plotting.computeKernel computeNystroem {self.path} {self.n_components}")
        C, C_diag = self.loadMatrices()
        np.fill_diagonal(C, C_diag[:self.n_components])
        # Compute W corresponding to the upper squared part of C
        W = C[:self.n_components, :self.n_components]
        u, sigma, vt = np.linalg.svd(W, full_matrices=False)
        u_k = u[:, :self.k]
        sigma_k = sigma[:self.k]
        vt_k = vt[:self.k, :]
        Wk = np.linalg.pinv(u_k @ np.diag(sigma_k) @ vt_k)
        return C @ Wk @ C.T
