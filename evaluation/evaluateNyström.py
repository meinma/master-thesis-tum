import os

import fire
import h5py

from matrix_factorization.nystroem import Nystroem
from plotting.createStartPlot import loadModel, loadDataset
from utils import solve_system_fast, loadTargets, oneHotEncoding, print_accuracy, load_kern

NYSTROM_PATH = "./evaluation/kernels.h5"


def evaluateNyström(fraction=0.2):
    model = loadModel()
    dataset = loadDataset()
    test = loadDataset(mode='test')
    val = loadDataset(mode='val')
    Y = loadTargets(dataset)
    os.system(f"python -m plotting.computeKernelComputeValidationAndTestKernel {NYSTROM_PATH}")
    components = int(fraction * len(dataset))
    nyström = Nystroem(components, k=None, dataset=dataset, model=model, path=NYSTROM_PATH)
    approximation = nyström.fit_transform()
    A = solve_system_fast(Kxx=approximation, Y=oneHotEncoding(Y))
    with h5py.File(NYSTROM_PATH, 'r') as f:
        Kxtx = load_kern(f['Kxtx'], 0)
        Kxvx = load_kern(f['Kxtx'], 0)
    Yt = loadTargets(test)
    Yv = loadTargets(val)
    print_accuracy(A, Kxvx, Yv, 'validation')
    print_accuracy(A, Kxtx, Yt, 'test')


if __name__ == "__main__":
    fire.Fire(evaluateNyström)
