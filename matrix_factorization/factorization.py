import absl.app
import matplotlib.pyplot as plt
import numpy as np
from fancyimpute import MatrixFactorization, IterativeSVD, SoftImpute
from sklearn import decomposition

FLAGS = absl.app.flags.FLAGS


def pca_analysis(x, k=None):
    PCA = decomposition.PCA(k)
    PCA.fit(x)
    eigenvectors = PCA.components_
    singular_values = PCA.singular_values_
    variance = PCA.explained_variance_
    print("Variance")
    print(variance)


def plotEigenvalues(x):
    """

    @param x: contains the matrix for which the eigenvalues are computed
    @return: plots the eigenvalues of the given matrix x
    """
    eigenvalues = np.linalg.eigvals(x)
    # x = np.arange(1, len(eigenvalues)+1, 1)
    fig = plt.figure()
    plt.plot(**eigenvalues)
    plt.xlabel("Number of eigenvalue")
    plt.ylabel("Eigenvalue")
    plt.title("Plot of the eigenvalues of the K_xx matrix")
    plt.show()


def matrix_completion(x):
    return MatrixFactorization().fit_transform(X=x)


def iterativeSVD(x):
    return IterativeSVD().fit_transform(X=x)


def softImpute(x):
    return SoftImpute().fit_transform(X=x)
