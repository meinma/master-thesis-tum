import os
from timeit import default_timer as timer

import fire
import h5py

from matrix_factorization.nystroem import Nystroem
from plotting.createStartPlot import loadDataset
from utils import solve_system_fast, loadTargets, oneHotEncoding, print_accuracy, load_kern, loadNormalizedModel

NYSTROM_PATH = "./evaluation/kernels.h5"
NYSTROM_PATH_EVAL = './evaluation/nyst.h5'


def evaluateNyström(fraction=0.2):
    start = timer()
    model = loadNormalizedModel()
    dataset = loadDataset()
    test = loadDataset(mode='test')
    val = loadDataset(mode='val')
    Y = loadTargets(dataset)
    os.system(f"python -m plotting.computeKernel computeValidationAndTestKernel {NYSTROM_PATH_EVAL}")
    components = int(fraction * len(dataset))
    nystroem = Nystroem(components, k=None, dataset=dataset, model=model, path=NYSTROM_PATH)
    approximation = nystroem.fit_transform()
    A = solve_system_fast(Kxx=approximation, Y=oneHotEncoding(Y))
    with h5py.File(NYSTROM_PATH_EVAL, 'r') as f:
        Kxvx = load_kern(f['Kxvx'], 0)
        Kxtx = load_kern(f['Kxtx'], 0)
    Yt = loadTargets(test)
    Yv = loadTargets(val)
    print_accuracy(A, Kxvx, Yv, 'validation')
    print_accuracy(A, Kxtx, Yt, 'test')
    end = timer()
    diff = (end - start) / 60
    print(diff)


if __name__ == "__main__":
    fire.Fire(evaluateNyström)
