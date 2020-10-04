"""
Given a pre-computed kernel and a data set, compute train/validation/test accuracy.
"""
import importlib

import absl.app
import h5py

from cnn_gp import DatasetFromConfig
from utils import load_kern, solve_system_old, diag_add, oneHotEncoding, print_accuracy

FLAGS = absl.app.flags.FLAGS


def main(_):
    config = importlib.import_module(f"configs.{FLAGS.config}")
    dataset = DatasetFromConfig(FLAGS.datasets_path, config)

    print("Reading training labels")
    _, Y = dataset.load_full(dataset.train)
    Y_1hot = oneHotEncoding(Y)

    with h5py.File(FLAGS.in_path, "r") as f:
        print("Loading kernel")
        Kxx = load_kern(f["Kxx"], 0)
        diag_add(Kxx, FLAGS.jitter)

        print("Solving Kxx^{-1} Y")
        A = solve_system_old(Kxx, Y_1hot)

        _, Yv = dataset.load_full(dataset.validation)
        Kxvx = load_kern(f["Kxvx"], 0)

        print_accuracy(A, Kxvx, Yv, "validation")
        del Kxvx
        del Yv

        _, Yt = dataset.load_full(dataset.test)
        Kxtx = load_kern(f["Kxtx"], 0)
        print_accuracy(A, Kxtx, Yt, "test")
        del Kxtx
        del Yt
        del Kxx


# @(py36) ag919@ulam:~/Programacio/cnn-gp-pytorch$ python classify_gp.py --in_path=/scratch/ag919/grams_pytorch/mnist_as_tf/00_nwork07.h5 --config=mnist_as_tf
# magma.py has some problem loading. Proceeding anyways using CPU.
# Original error: ignoring magma shit
# Reading training labels
# Loading kernel
# Solving Kxx^{-1} Y
# Running scipy solve Kxx^-1 Y routine
# train accuracy: 10.26%
# validation accuracy: 99.31%
# test accuracy: 99.11999999999999%


if __name__ == '__main__':
    f = absl.app.flags
    f.DEFINE_string("datasets_path", "/scratch/ag919/datasets/",
                    "where to save datasets")
    f.DEFINE_string("config", "mnist", "which config to load from `configs`")
    f.DEFINE_string('in_path', "/scratch/ag919/grams_pytorch/mnist/dest.h5",
                    "path of h5 file to load kernels from")
    f.DEFINE_float("jitter", 0.0, "add to the diagonal")
    f.DEFINE_float("computation", 1.0, "fraction of kxx which is computed exactly")
    absl.app.run(main)
