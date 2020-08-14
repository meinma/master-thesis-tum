import h5py
import numpy as np
import torch
from torch.utils.data import Subset

from cnn_gp.kernel_save_tools import save_K
from utils.utils import load_kern


class Nystroem:
    def __init__(self, n_components, k, dataset, model, batch_size, out_path, worker_rank=0, n_workers=1,
                 in_path=None, same=True, diag=False):
        if in_path is None:
            self.in_path = out_path
        else:
            self.in_path = in_path
        self.n_components = n_components
        self.model = model
        if k is None:
            self.k = self.n_components
        else:
            self.k = k
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.worker_rank = worker_rank
        self.out_path = out_path
        self.same = same
        self.diag = diag
        self.indices = None
        self.subset = None
        self.sample()

    def sample(self):
        """
        Sample m data points for which the kernel is evaluated exactly
        """
        self.indices = list(range(self.n_components))
        self.subset = Subset(self.dataset, self.indices)

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
            # rectangular matrix consisting of W and S
            save_K(f, kern, name="C", X=self.dataset, X2=self.subset, diag=False, **kwargs)
            #  Compute the diagonal for C for better accuracy
            save_K(f, kern, name="Cd", X=self.dataset, X2=None, diag=True, **kwargs)

    def loadMatrices(self):
        """

        @return: the matrices W and C which have been computed before and stored as h5py file
        """
        with h5py.File(self.in_path, "r") as f:
            print("Loading kernel")
            C = load_kern(f["C"], 0)
            C_diag = load_kern(f["Cd"], 0, diag=True)
        return C.numpy(), C_diag.numpy()

    def fit_transform(self):
        """
        @return: the approximation of the original kernel matrix corresponding to
        G_k = C@W_k^+@C.T
        """
        self.computeKernelMatrices()
        C, C_diag = self.loadMatrices()
        np.fill_diagonal(C, C_diag[:self.n_components])
        W = C[:self.n_components, :self.n_components]
        u, sigma, vt = np.linalg.svd(W, full_matrices=False)
        u_k = u[:, :self.k]
        sigma_k = sigma[:self.k]
        vt_k = vt[:self.k, :]
        Wk = np.linalg.pinv(u_k @ np.diag(sigma_k) @ vt_k)
        return C @ Wk @ C.T
