import absl.app
from fancyimpute import MatrixFactorization
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


def matrix_completion(x):
    return MatrixFactorization().fit_transform(X=x)
