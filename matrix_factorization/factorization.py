import fire
import seaborn as sns
from fancyimpute import MatrixFactorization, IterativeSVD, SoftImpute

sns.set()


def matrix_completion(x):
    return MatrixFactorization().fit_transform(X=x)


def iterativeSVD(x):
    return IterativeSVD().fit_transform(X=x)


def softImpute(x):
    return SoftImpute().fit_transform(X=x)


if __name__ == "__main__":
    fire.Fire()
