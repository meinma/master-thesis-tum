import fire
import h5py
import torch
from torch.utils.data import Subset

from cnn_gp import save_K
from plotting.createStartPlot import loadDataset, loadModel
from utils import load_kern, constructSymmetricMatrix


def computeKxxMatrix(path, name):
    """
    Stores the kernel matrix Kxx between the training points
    @param path: file path containing the kernel matrix
    @param name: name of the dataset of the kernel matrix
    @return:
    """
    model = loadModel()
    dataset = loadDataset()
    kwargs = dict(worker_rank=0, n_workers=1,
                  batch_size=200, print_interval=2.)

    def kern(x, x2, **args):
        with torch.no_grad():
            return model(x.cuda(), x2.cuda(), **args).detach().cpu().numpy()

    with h5py.File(path, "w") as f:
        save_K(f, kern, name, X=dataset, X2=None, diag=False, **kwargs)


def computeNystroem(path, components):
    model = loadModel()
    dataset = loadDataset()

    def kern(x, x2, **args):
        with torch.no_grad():
            return model(x.cuda(), x2.cuda(), **args).detach().cpu().numpy()

    with h5py.File(path, "w") as f:
        kwargs = dict(worker_rank=0, n_workers=1,
                      batch_size=200, print_interval=2.)

        save_K(f, kern, name="C", X=dataset, X2=Subset(dataset, range(int(components))), diag=False, **kwargs)
        save_K(f, kern, name="Cd", X=dataset, X2=None, diag=True, **kwargs)
    print("Compute Nystroem done")


def loadMatrixFromDiskAndMirror(path, name):
    """
    Loads a matrix from a file and mirrors it if desired before returning
    @param path: File path containing the matrix
    @param name: Name of the dataset in the file
    """
    with h5py.File(path, "a") as f:
        matrix = load_kern(f[name], 0)
        sym_Matrix = constructSymmetricMatrix(matrix)
        del f[name]
        f.create_dataset(name=name, shape=(25000, 25000), data=sym_Matrix)


if __name__ == "__main__":
    fire.Fire()
