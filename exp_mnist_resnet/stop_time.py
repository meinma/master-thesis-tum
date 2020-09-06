from timeit import default_timer as timer

import fire
import h5py
import numpy as np


def startTimer():
    print("Timer is starting")
    start = timer()
    hf = h5py.File('time.h5', 'w')
    hf.create_dataset('timing', data=start)
    hf.close()


def endTimer():
    print("Timer ends")
    end = timer()
    hf = h5py.File('time.h5', 'r')
    start = hf.get('timing')
    start = np.array(start)
    hf.close()
    diff = (end - start) // 60
    print("Run time:")
    print(diff)


if __name__ == "__main__":
    fire.Fire()
