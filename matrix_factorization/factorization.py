import absl.app
import fire
import matplotlib.pyplot as plt
import numpy as np
from fancyimpute import MatrixFactorization, IterativeSVD, SoftImpute
from sklearn import kernel_approximation

FLAGS = absl.app.flags.FLAGS


def applyNyström(x, components):
    return kernel_approximation.Nystroem().fit_transform(x, n_components=components)


def plotEigenvalues(x: np.ndarray):
    """
    @param x: contains the matrix for which the eigenvalues are computed
    @return: plots the eigenvalues of the given matrix x
    """
    print("Plotting eigenvalues")
    eigenvalues = np.linalg.eigvals(x)
    print(eigenvalues)
    plt.figure()
    plt.plot(eigenvalues)
    plt.xlabel("Number of eigenvalue")
    plt.ylabel("Eigenvalue")
    plt.title("Plot of the eigenvalues of the K_xx matrix")
    plt.show()
    plt.savefig('./plots/eigenvalues.svg')


def matrix_completion(x):
    return MatrixFactorization().fit_transform(X=x)


def iterativeSVD(x):
    return IterativeSVD().fit_transform(X=x)


def softImpute(x):
    return SoftImpute().fit_transform(X=x)


def testPlot():
    plt.figure()
    plt.plot([1, 2, 3, 4])
    plt.title("Testplot")
    plt.show()
    print("Plot finished")
    plt.savefig('./plots/testplot.svg')


if __name__ == "__main__":
    fire.Fire()
