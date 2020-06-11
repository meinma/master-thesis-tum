import absl.app
import h5py

from exp_mnist_resnet.classify_gp import load_kern

FLAGS = absl.app.flags.FLAGS


def pca_analysis(x, k=None):
    print(x)


def main():
    with h5py.File(FLAGS.in_path, "r") as f:
        print("Loading kernel")
        Kxx = load_kern(f["Kxx"], 0)
