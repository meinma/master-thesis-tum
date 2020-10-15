# Code repository belonging to the Master's Thesis "Efficient Implementation of Deep Convolutional Gaussian Processes" by Martin Meinel
The thesis builds up on the paper of 
```bibtex
@inproceedings{aga2018cnngp,
  author    = {{Garriga-Alonso}, Adri{\`a} and Aitchison, Laurence and Rasmussen, Carl Edward},
  title     = {Deep Convolutional Networks as shallow {G}aussian Processes},
  booktitle = {International Conference on Learning Representations},
  year      = {2019},
  url       = {https://openreview.net/forum?id=Bklfsi0cKm}}
```
and therefore also uses the code of them publicly available on their own git: https://github.com/cambridge-mlg/cnn-gp.

In general only the configuration with the ConvNet GP and MNIST is used and changed from the original version. 
So, all configurations can be adjusted in the config file of ConvNet GP [ConvNetGP Config](/configs/mnist_paper_convnet_gp.py).
Here can the training, validation and test be set.

All libary dependencies are listed in the [requirements file](requirements.txt).
It is necessary to have one GPU. Otherwise the code has to be adjusted and Matrix Factorization cannot be used.

# Replicating the experiments
Obviously, the experiment results are strongly dependent on the used machine, since the efficiency is measured and cannot be reproduced exactly on any other machine. But the experiments can be ran on any machine with a GPU.

To run each of the
experiments, first take a look at the file `exp_mnist_resnet/run.bash`. Edit the configuration variables near the top
appropriately. Then, run it from the root of the directory, like that:

```bash
bash ./exp_mnist_resnet/run.bash
```

As it is now the original ConvNet GP is ran and will be evaluated on the previously defined validation and test set. Additionally, the total time is measured.

By replacing the code after the configuration in `exp_mnist_resnet/run.bash` by one of the following snippets the single experiments can be replicated. 
- A plot is created showing that the run time of the ConvNet GP kernel increases quadratically if the training points increase linearly.
```bash
 python -m plotting.createStartPlot
 ``` 
- A random 5000 x 5000 matrix is created and then randomly 10% up to 90% of the values are omitted and approximated by Iterative SVD, Soft Impute and MAtrix Factorization.
The mean fo the Relative RMSE, its variance and the time it takes to approximate is assessed over several runs and plotted. It also generates plots of the minimum, maximum and medium error of the approximation and the original random matrix.
```bash
python -m matrix_factorization.experiments
```
- For the 5000 x 5000 only some part is computed exactly with the model makinguse of the normalization layer. 
The other part is approximated by Iterative SVD, Soft Impute, Matrix Factorization and the Nystöm method. Analogous to the previous experiment it changes from approximating 90% to 10% of the matrix.
The Relative RMSE in the log space, the time it takes for the approximation and the accuracy which is obtained by using the approximated matrix for generating predictions is plotted over the fraction of the matrix elements, which are given.
This code can be adjusted so that also the results for the 12 500 x 12 500 matrix and using only Iterative SVD and the Nyström method can be obtained.
````bash
python -m plotting.compareMethodsPlot compareAccuracyOverTime 
````
 - For any training, validation and test set Iterative SVD can be evaluated with following 
 By default it computes only 20% of the kernel matrix exactly. This can be adjusted [here.](/evaluation/evaluateIterativeSVD.py)
 The evaluation also measures the time and is started by using:
 ```bash
python -m evaluation.evaluateIterativeSVD
```
 - Analogous to the evaluation with Iterative SVD also the Nyström method can be used. The amount of training samples for which the kernel matrix is evaluated exactly can be changed [here.](/evaluation/evaluateNyström.py)
 The evaluation can be called by putting:
 ```bash
python -m evaluation.evaluateNyström
```
# Architecture of the original and ConvNet GP and the one with the Normalization layer
<details>
  <summary>(click to expand) Architecture for ConvNet GP</summary>

  ```python
from cnn_gp import Sequential, Conv2d, ReLU
var_bias = 7.86
var_weight = 2.79

initial_model = Sequential(
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      ReLU(),  # Total 7 layers before dense

      Conv2d(kernel_size=28, padding=0, var_weight=var_weight, var_bias=var_bias),
  ```
</details>

<details>
  <summary>(click to expand) Architecture for ConvNet GP with Normalization Layer</summary>

  ```python
from cnn_gp import Sequential, Conv2d, ReLU, NormalizationModule
var_bias = 7.86
var_weight = 2.79

initial_model = Sequential(
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      NormalizationModule(),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      NormalizationModule(),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      NormalizationModule(),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      NormalizationModule(),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      NormalizationModule(),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      NormalizationModule(),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      NormalizationModule(),
      ReLU(),  # Total 7 layers before dense

      Conv2d(kernel_size=28, padding=0, var_weight=var_weight, var_bias=var_bias),
  ```
</details>




The code might seem unconventional at the beginning, since there exist a lot of calls from the terminal. However, by doing it that way less RAM is used, which is important for big data sets.

