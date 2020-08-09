import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fancyimpute import MatrixFactorization, IterativeSVD, SoftImpute

sns.set()


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
    plt.savefig('./plots/eigenvalues25.svg')
    plt.close()
    plt.figure()
    plt.plot(np.log(eigenvalues))
    plt.xlabel('Number of the eigenvalue')
    plt.ylabel('Log space of eigenvalue')
    plt.title('Plot of the eigenvalues of the K_xx matrix in the log space')
    plt.show()
    plt.savefig('./plots/log_eigenvalues.svg')


def matrix_completion(x):
    return MatrixFactorization().fit_transform(X=x)


def iterativeSVD(x):
    return IterativeSVD().fit_transform(X=x)


def softImpute(x):
    return SoftImpute().fit_transform(X=x)


if __name__ == "__main__":
    fire.Fire()
