import os
from timeit import default_timer as timer

import fire
import h5py
import numpy as np

from plotting.createStartPlot import loadDataset
from utils import load_kern, loadTargets, solve_system_fast, oneHotEncoding, print_accuracy

SVD_PATH = "./evaluation/svd.h5"
SOL_PATH = "./evaluation/sol.h5"
SVD_PATH_EVAL = './evaluation/eval.h5'


def evaluateIterSVD(fraction=0.2):
    start = timer()
    os.system(f"python -m plotting.computeKernel computeValidationAndTestKernel {SVD_PATH_EVAL}")
    os.system(f"python -m plotting.computeKernel computeKxxMatrix {SVD_PATH} Kxx_pert {fraction}")
    os.system(f"python -m plotting.computeMatrixFactorization {SVD_PATH} {SOL_PATH} svd")
    with h5py.File(SOL_PATH, 'r') as f:
        approx = np.array(f.get('approx'))
    with h5py.File(SVD_PATH_EVAL, 'r') as f:
        Kxvx = load_kern(f['Kxvx'], 0)
        Kxtx = load_kern(f['Kxtx'], 0)
    dataset = loadDataset()
    val = loadDataset(mode='val')
    test = loadDataset(mode='test')
    Y = loadTargets(dataset)
    Yv = loadTargets(val)
    Yt = loadTargets(test)
    A = solve_system_fast(approx, oneHotEncoding(Y))
    print_accuracy(A, Kxvx, Yv, 'validation')
    print_accuracy(A, Kxtx, Yt, 'test')
    end = timer()
    diff = (end - start) / 60
    print(diff)


if __name__ == "__main__":
    fire.Fire(evaluateIterSVD)
