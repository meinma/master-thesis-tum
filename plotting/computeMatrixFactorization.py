from timeit import default_timer as timer

import fire
import h5py
import numpy as np
from fancyimpute import MatrixFactorization, IterativeSVD, SoftImpute


def solve(inpath, outpath, mode):
    """
    Reads matrix which is supposed to be approximated from h5 file and stores time and approximation back
    @param mode: Dtermines the type of approximation ('mf', 'svd' or 'soft')
    @param outpath: stores the result (time and approximation there)
    @param inpath: location of matrix which is supposed to be approximated
    @return:
    """
    if mode == 'mf':
        solver = MatrixFactorization(epochs=1000, min_improvement=0.01)
    elif mode == 'svd':
        solver = IterativeSVD()
    else:
        solver = SoftImpute()
    with h5py.File(inpath, 'r') as f:
        Kxx_pert = np.array(f.get('Kxx_pert'))
        # f.close()
    start = timer()
    approx = solver.fit_transform(Kxx_pert)
    end = timer()
    diff = (end - start) // 60
    with h5py.File(outpath, 'w') as f:
        f.create_dataset(name='time', data=diff)
        f.create_dataset(name="approx", shape=(Kxx_pert.shape[0], Kxx_pert.shape[1]), data=approx)
        f.close()


if __name__ == "__main__":
    fire.Fire(solve)
