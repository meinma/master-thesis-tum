import random

import h5py
import numpy as np
import torch
from torch.utils.data import Subset

from cnn_gp.kernel_save_tools import save_K
from utils.utils import load_kern, constructSymmetricIfNotSymmetric


class Nystroem:
    def __init__(self, n_components, k, dataset, model, batch_size, out_path, worker_rank=0, n_workers=1,
                 in_path=None, same=True, diag=False):
        if in_path is None:
            self.in_path = out_path
        else:
            self.in_path = in_path
        self.n_components = n_components
        self.model = model
        self.k = k
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.worker_rank = worker_rank
        self.out_path = out_path
        self.same = same
        self.diag = diag
        self.subset = self.sample()

    def sample(self) -> Subset:
        """
        Sample m data points for which the kernel is evaluated exactly
        @return: Subset of m data points of the original data set
        """
        n_samples = range(len(self.dataset))
        indices = random.sample(n_samples, self.n_components)
        return Subset(self.dataset, indices)

    def computeKernelMatrices(self):
        """
        @return:
        """

        def kern(x, x2, **args):
            with torch.no_grad():
                return self.model(x.cuda(), x2.cuda(), **args).detach().cpu().numpy()

        with h5py.File(self.out_path, "w") as f:
            kwargs = dict(worker_rank=self.worker_rank, n_workers=self.n_workers,
                          batch_size=self.batch_size, print_interval=2.)
            save_K(f, kern, name="W", X=self.subset, X2=None, diag=False, **kwargs)
            save_K(f, kern, name="C", X=self.dataset, X2=self.subset, diag=False, **kwargs)

    def loadMatrices(self):
        """

        @return: the matrices W and C which have been computed before and stored as h5py file
        """
        with h5py.File(self.in_path, "r") as f:
            print("Loading kernel")
            W = load_kern(f["W"], 0)
            C = load_kern(f["C"], 0)
        return W.numpy(), C.numpy()

    def fit_transform(self):
        """
        @return: the approximation of the original kernel matrix corresponding to
        G_k = C@W_k^+@C.T
        """
        # 1. #Compute C and W
        self.computeKernelMatrices()
        # 2.
        W, C = self.loadMatrices()
        W2 = C[:3500, :3500]
        close = np.isclose(W, W2)
        print(f"W is equal to W2: {close}")
        W = constructSymmetricIfNotSymmetric(W)
        print(f"Shape of W: {W.shape}")
        print(f"Shape of C: {C.shape}")
        u, sigma, vt = np.linalg.svd(W, full_matrices=False)
        u_k = u[:, :self.k]
        sigma_k = sigma[:self.k]
        vt_k = vt[:self.k, :]
        Wk = np.linalg.pinv(u_k @ np.diag(sigma_k) @ vt_k)
        return C @ Wk @ C.T
