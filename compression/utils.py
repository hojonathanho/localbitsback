import math
import random

import numpy as np
import torch
import torch.utils.data
import torchvision


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup(seed):
    seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_printoptions(precision=5, linewidth=1000)


def sumflat(x: torch.Tensor):
    return x.view(x.shape[0], -1).sum(1)


def standard_normal_logp(x):
    return -0.5 * x ** 2 - 0.5 * math.log(math.tau)


def inverse_sigmoid(x: torch.Tensor):
    return torch.log(x) - torch.log1p(-x)


def process_gaussian(A: torch.Tensor, scale: float, inverted: bool):
    """
    Convert a Gaussian x = scale*A z, with z ~ N(0, I), into a linear autoregressive model
    Calling this function with (A, scale=s) is equivalent to calling it with (s*A, scale=1),
    except with possibly better numerical stability.
    """
    assert isinstance(A, torch.Tensor) and len(A.shape) == 2 and A.shape[0] == A.shape[1]
    dim = A.shape[0]
    if inverted:
        # Here, the goal is to convert x = A^{-1} z into a linear AR model.
        # If R (upper triangular) comes from the QR decomposition A, then
        # R^T R = A^T A, so A^{-1} A^{-1}^T = R^{-1} R^{-1}^T,
        # which implies that R^{-1} z has the same distribution as A^{-1} z.
        # So, if x solves the equation R x = z, then x will have the distribution we desire.
        # Because R is triangular, x can be obtained via back substitution, which is equivalent to sampling from
        # a linear AR model driven by Gaussian noise z. The coefficients of the linear functions that
        # define the conditionals of this AR model can be read off in the rows of R; the ordering of the
        # variables goes backwards because R is upper triangular.
        _, R = torch.qr(A)
        stds = (1.0 / scale) / R.diag().abs()  # standard deviations of conditionals
        R /= -R.diag()[:, None]  # coefficients of linear functions defining the means
        mean_coefs = R
    else:
        # Here, the goal is to convert x = A z into a linear AR model
        L = torch.qr(A.t())[1].t()  # Cholesky decomposition of AA^T
        stds = L.diag().abs() * scale  # standard deviations of conditionals
        Linv, _ = torch.triangular_solve(torch.eye(dim, dtype=L.dtype, device=L.device), L, upper=False)  # invert L
        Linv *= -L.diag()[:, None]
        mean_coefs = Linv
    mean_coefs[range(dim), range(dim)] = 0  # set diagonal to zero; AR conditionals don't depend on current timestep
    return mean_coefs, stds


# Dataset utilities

class CIFAR10WithoutLabels(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        return (super().__getitem__(index)[0],)


def load_imagenet_data(npy_path):
    """npy_path should be the path to a file created by the create_imagenet_benchmark_datasets.py script"""
    return torch.utils.data.TensorDataset(torch.from_numpy(np.load(npy_path)).permute(0, 3, 1, 2).to(dtype=torch.int64))


def make_testing_dataloader(dataset, *, seed, limit_dataset_size, bs):
    # Fixed random permutation of the dataset for experiment purposes only
    dataset = torch.utils.data.dataset.Subset(
        dataset,
        np.random.RandomState(seed=seed).permutation(len(dataset)).tolist()
    )
    # Subsample dataset if requested
    if limit_dataset_size:
        dataset = torch.utils.data.dataset.Subset(dataset, list(range(0, min(len(dataset), limit_dataset_size))))
    # Turn into batches
    return torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False), dataset
