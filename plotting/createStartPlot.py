from timeit import default_timer as timer

import fire
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import torch
from torch.utils.data import Subset
from tqdm import tqdm

from cnn_gp import DatasetFromConfig
from cnn_gp import save_K
from configs import mnist_paper_convnet_gp
from utils import load_kern, deleteDataset

sn.set()


def warmUp_GPU():
    model = loadModel()
    args = {"same": False, "diag": False}
    for _ in range(5):
        x1 = torch.randn((5, 1, 28, 28)).cuda()
        x2 = torch.randn((4, 1, 28, 28)).cuda()
        _ = model(x1, x2, **args)


def loadModel(config=mnist_paper_convnet_gp):
    """
    Loads the model from the given config
    @return: model on GPU
    """
    return config.initial_model.cuda()


def loadDataset(path="./scratch/datasets/", config=mnist_paper_convnet_gp):
    """
    Returns the entire dataset specified by its path and the configuration (train_range, validation_range..)
    @param path: dataset path
    @param config: config
    @return:
    """
    return DatasetFromConfig(path, config=config).train


def sample_data(dataset, samples):
    """
    Takes random samples of the given dataset
    @param dataset: specifies dataset
    @param samples: specifies amount of samples
    @return: subset of samples of the given dataset
    """
    indices = np.random.randint(0, len(dataset), samples)
    return Subset(dataset, indices)


def actionToMeasure(model, data, path):
    """
    This method contains all function calls which are supposed to be measured
    1. Computing kernel
    2. Loading kernel
    @return:
    """

    def kern(x, x2, **args):
        with torch.no_grad():
            return model(x.cuda(), x2.cuda(), **args).detach().cpu().numpy()

    """"Create file for kernel matrix and use only one worker no parallelization"""
    with h5py.File(path, "a") as f:
        kwargs = dict(worker_rank=0, n_workers=1,
                      batch_size=200, print_interval=2.)
        save_K(f, kern, name="Kxx", X=data, X2=None, diag=False, **kwargs)
        """The loading process is included in the time measurement on purpose.
        Otherwise we could not compare it later on to methods where only a 
        part is computed and the rest is approximated, since the kernel matrix has to be loaded in any
        case
        """
        _ = load_kern(f['Kxx'], 0)


def plotTimes():
    path = './plotting/timing.h5'
    kernelSizes = [200, 5000, 10000, 15000, 20000, 25000]
    repetitions = 5
    dataset = loadDataset()
    model = loadModel()
    timings = np.zeros((len(kernelSizes), repetitions))
    warmUp_GPU()
    for index, size in tqdm(enumerate(kernelSizes)):
        for i in tqdm(range(repetitions)):
            data = sample_data(dataset, size)
            start = timer()  # CPU measurement
            actionToMeasure(model=model, data=data, path=path)
            end = timer()
            diff = (end - start) // 60  # CPU time
            print(f"Curr time: {diff}")
            timings[index][i] = diff
            deleteDataset(path)
    means = np.mean(timings, axis=1)
    # Fit a polynomial to the data
    parameters = np.polyfit(kernelSizes, means, deg=2)
    polynomial = np.poly1d(parameters)
    plt.title('Time to compute the kernel matrix over the amount of data points')
    plt.xlabel('Size of the kernel matrix')
    plt.ylabel('Computational time in minutes')
    plt.plot(kernelSizes, means, label="Original Plot")
    plt.plot(kernelSizes, polynomial(kernelSizes), label="Fitted polynomial of degree 2")
    plt.show()
    plt.savefig('./plots/beginnningPlot.svg')
    plt.close()


if __name__ == "__main__":
    fire.Fire(plotTimes)
