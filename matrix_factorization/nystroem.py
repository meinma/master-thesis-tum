import torch
from sklearn.kernel_approximation import Nystroem
from torch.utils.data import DataLoader


class NystroemApproximator:
    def __init__(self, model, n_components=10, same=True, diag=False):
        self.model = model
        self.n_components = n_components
        self.diag = diag
        self.same = same

    def getModelArguments(self):
        arguments = {'same': self.same, 'diag': self.diag}
        return arguments

    def fit_transform(self, data):
        """The scikit learn method only allows data in the form (samples, features) as numpy.ndarrays.
                This is why the following code takes the trainings set and transforms it to a flattened numpy array.
                In order to make the model work with the flattened numpy array the model input is transformed
                back to its original shape (batch_size, 1, 28, 28) and to a pyTorch tensor.
                This is also helpful due to the fact that the GPU can be used again"""

        def kern(x, x2, **kwargs):
            with torch.no_grad():
                return self.model(torch.from_numpy(x.reshape(1, 1, 28, 28)).cuda(),
                                  torch.from_numpy(x2.reshape(1, 1, 28, 28)).cuda(), **kwargs) \
                    .detach().cpu().numpy().item()

        size = len(data)
        trainLoader = DataLoader(data, batch_size=size)
        data = next(iter(trainLoader))[0].numpy().reshape(size, -1)
        print(data.shape)
        nystroem_map = Nystroem(kernel=kern, kernel_params=self.getModelArguments(), n_components=self.n_components)
        return nystroem_map.fit_transform(data)
