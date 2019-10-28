"""
Utilties for logistic and mixture of logistic distributions
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F


def logistic_logpdf(x, *, mean, logscale):
    """
    log density of logistic distribution
    """
    z = (x - mean) * torch.exp(-logscale)
    return z - logscale - 2 * F.softplus(z)


def logistic_logcdf(x, *, mean, logscale):
    """
    log cdf of logistic distribution
    """
    return F.logsigmoid((x - mean) * torch.exp(-logscale))


@torch.no_grad()
def test_logistic():
    import numpy as np
    import scipy.stats

    # Evaluate log pdf at these points
    n = 100
    xs = torch.linspace(-5, 5, n, dtype=torch.float64)

    # Test against scipy
    for loc in np.linspace(-1, 2, 5):
        means = torch.DoubleTensor([loc] * n)
        for scale in np.linspace(.01, 3, 5):
            logscales = torch.log(torch.DoubleTensor([scale] * n))
            true_logpdfs = scipy.stats.logistic.logpdf(xs.numpy(), loc, scale)
            true_logcdfs = scipy.stats.logistic.logcdf(xs.numpy(), loc, scale)
            logpdfs = logistic_logpdf(xs, mean=means, logscale=logscales)
            logcdfs = logistic_logcdf(xs, mean=means, logscale=logscales)
            assert np.allclose(logpdfs.numpy(), true_logpdfs)
            assert np.allclose(logcdfs.numpy(), true_logcdfs)


def mixlogistic_logpdf(x, *, logits, means, logscales, mix_dim):
    """
    logpdf of a mixture of logistics
    dim: the axis for indexing into mixture components
    """
    assert len(x.shape) + 1 == len(logits.shape) and logits.shape == means.shape == logscales.shape

    mix_logprobs = F.log_softmax(logits, dim=mix_dim)
    assert mix_logprobs.shape == logits.shape

    logpdfs = logistic_logpdf(x.unsqueeze(dim=mix_dim), mean=means, logscale=logscales)
    assert len(logpdfs.shape) == len(logits.shape)

    out = torch.logsumexp(mix_logprobs + logpdfs, dim=mix_dim)
    assert out.shape == x.shape
    return out


def mixlogistic_logcdf(x, *, logits, means, logscales, mix_dim):
    """
    logcdf of a mixture of logistics
    dim: the axis for indexing into mixture components
    """
    assert len(x.shape) + 1 == len(logits.shape) and logits.shape == means.shape == logscales.shape

    mix_logprobs = F.log_softmax(logits, dim=mix_dim)
    assert mix_logprobs.shape == logits.shape

    logcdfs = logistic_logcdf(x.unsqueeze(dim=mix_dim), mean=means, logscale=logscales)
    assert len(logcdfs.shape) == len(logits.shape)

    out = torch.logsumexp(mix_logprobs + logcdfs, dim=mix_dim)
    assert out.shape == x.shape
    return out


@torch.no_grad()
def test_logistic_mixture():
    import numpy as np
    import scipy.stats

    n = 100
    xs = torch.linspace(-5, 5, n, dtype=torch.float64)
    logits = [.1, .2, 4]
    means = [-1., 0., 1]
    logscales = [-5., 0., 0.2]

    logpdfs = mixlogistic_logpdf(
        xs,
        logits=torch.DoubleTensor([logits] * n),
        means=torch.DoubleTensor([means] * n),
        logscales=torch.DoubleTensor([logscales] * n),
        mix_dim=-1
    )
    logcdfs = mixlogistic_logcdf(
        xs,
        logits=torch.DoubleTensor([logits] * n),
        means=torch.DoubleTensor([means] * n),
        logscales=torch.DoubleTensor([logscales] * n),
        mix_dim=-1
    )

    probs = np.exp(logits) / np.exp(logits).sum()
    scipy_probs = 0.
    scipy_cdfs = 0.
    for p, m, ls in zip(probs, means, logscales):
        scipy_probs += p * scipy.stats.logistic.pdf(xs, m, np.exp(ls))
        scipy_cdfs += p * scipy.stats.logistic.cdf(xs, m, np.exp(ls))

    assert scipy_probs.shape == logpdfs.shape
    assert np.allclose(logpdfs, np.log(scipy_probs))
    assert np.allclose(logcdfs, np.log(scipy_cdfs))


_FORCE_ACCURATE_INV_CDF = False


@contextmanager
def force_accurate_mixlogistic_invcdf():
    global _FORCE_ACCURATE_INV_CDF
    prev_val = _FORCE_ACCURATE_INV_CDF
    _FORCE_ACCURATE_INV_CDF = True
    yield
    _FORCE_ACCURATE_INV_CDF = prev_val


@torch.no_grad()
def mixlogistic_invcdf(y, *, logits, means, logscales, mix_dim,
                       tol=1e-8, max_bisection_iters=60, init_bounds_scale=100.):
    """
    inverse cumulative distribution function of a mixture of logistics, via bisection
    """
    if _FORCE_ACCURATE_INV_CDF:
        tol = min(tol, 1e-14)
        max_bisection_iters = max(max_bisection_iters, 200)
        init_bounds_scale = max(init_bounds_scale, 100.)
    return mixlogistic_invlogcdf(y.log(), logits=logits, means=means, logscales=logscales, mix_dim=mix_dim,
                                 tol=tol, max_bisection_iters=max_bisection_iters, init_bounds_scale=init_bounds_scale)


@torch.no_grad()
def test_mixlogistic_invcdf():
    n = 100
    range_max = 30
    xs = torch.linspace(-range_max, range_max, n, dtype=torch.float64)
    logits = torch.DoubleTensor([[.1, .2, 4]] * n)
    means = torch.DoubleTensor([[-1., 0., 1]] * n)
    logscales = torch.DoubleTensor([[-5., 0., 0.2]] * n)
    logistic_args = dict(logits=logits, means=means, logscales=logscales, mix_dim=-1)
    out_logcdf = mixlogistic_logcdf(xs, **logistic_args)
    out_inv_cdf = mixlogistic_invcdf(torch.exp(out_logcdf), **logistic_args)
    assert out_inv_cdf.shape == xs.shape
    err = (out_inv_cdf - xs).abs().max()
    print('err', err)
    assert err < 1e-5


@torch.no_grad()
def mixlogistic_invlogcdf(log_y, *, logits, means, logscales, mix_dim,
                          tol, max_bisection_iters, init_bounds_scale):
    """
    inverse log cumulative distribution function of a mixture of logistics, via bisection
    """
    # import time
    # tstart = time.time()
    assert len(log_y.shape) + 1 == len(logits.shape) == len(means.shape) == len(logscales.shape)
    # assert (y >= 0).all() and (y <= 1).all()
    # zero initial guess
    x = torch.zeros_like(log_y)
    # initial search bounds: start far from the mean
    maxscales = logscales.exp().sum(dim=mix_dim, keepdim=True)  # sum of scales across mixture components
    lb, _ = (means - init_bounds_scale * maxscales).min(dim=mix_dim)
    ub, _ = (means + init_bounds_scale * maxscales).max(dim=mix_dim)
    # bisection
    for _iter in range(max_bisection_iters):
        cur_log_y = mixlogistic_logcdf(x, logits=logits, means=means, logscales=logscales, mix_dim=mix_dim)
        gt = cur_log_y > log_y
        new_x = torch.where(gt, (x + lb) / 2., (x + ub) / 2.)
        assert new_x.shape == x.shape == log_y.shape == gt.shape
        cur_err = (cur_log_y - log_y).abs().max().item()
        if cur_err < tol:
            x = new_x
            break
        lb = torch.where(gt, lb, x)
        ub = torch.where(gt, x, ub)
        x = new_x
    if cur_err > tol:
        print(
            'Warning: logistic mixture CDF inversion achieved error {} in {} iterations, instead of the desired tolerance {}'.format(
                cur_err, _iter, tol
            ))
    # print(_iter, (cur_log_y - log_y).abs().max().item(), (cur_log_y.exp() - log_y.exp()).abs().max().item())#,
    # time.time() - tstart)
    return x


@torch.no_grad()
def test_mixlogistic_invlogcdf():
    n = 100
    range_max = 30
    xs = torch.linspace(-range_max, range_max, n, dtype=torch.float64)
    logits = torch.DoubleTensor([[.1, .2, 4]] * n)
    means = torch.DoubleTensor([[-1., 0., 1]] * n)
    logscales = torch.DoubleTensor([[-5., 0., 0.2]] * n)
    logistic_args = dict(logits=logits, means=means, logscales=logscales, mix_dim=-1)
    out_logcdf = mixlogistic_logcdf(xs, **logistic_args)
    out_inv_cdf = mixlogistic_invlogcdf(out_logcdf, **logistic_args)
    assert out_inv_cdf.shape == xs.shape
    err = (out_inv_cdf - xs).abs().max()
    print('logspace err', (out_inv_cdf - xs).abs())
    assert err < 1e-5
