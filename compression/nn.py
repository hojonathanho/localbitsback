from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn import init

_WN_INIT_STDV = 0.05
_SMALL = 1e-10

_INIT_ENABLED = False


def is_init_enabled():
    return _INIT_ENABLED


@contextmanager
def init_mode():
    global _INIT_ENABLED
    assert not _INIT_ENABLED
    _INIT_ENABLED = True
    yield
    _INIT_ENABLED = False


class DataDepInitModule(Module):
    """
    Module with data-dependent initialization
    """

    def __init__(self):
        super().__init__()
        # self._wn_initialized = False

    def _init(self, *args, **kwargs):
        """
        Data-dependent initialization. Will be called on the first forward()
        """
        raise NotImplementedError

    def _forward(self, *args, **kwargs):
        """
        The standard forward pass
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Calls _init (with no_grad) if not initialized.
        If initialized already, calls _forward.
        """
        # assert self._wn_initialized == (not _INIT_ENABLED)
        if _INIT_ENABLED:  # not self._wn_initialized:
            # self._wn_initialized = True
            with torch.no_grad():  # no gradients for the init pass
                return self._init(*args, **kwargs)
        return self._forward(*args, **kwargs)


class Dense(DataDepInitModule):
    def __init__(self, in_features, out_features, init_scale=1.0):
        super().__init__()
        self.in_features, self.out_features, self.init_scale = in_features, out_features, init_scale

        self.w = Parameter(torch.Tensor(out_features, in_features))
        self.b = Parameter(torch.Tensor(out_features))

        init.normal_(self.w, 0, _WN_INIT_STDV)
        init.zeros_(self.b)

    def _init(self, x):
        y = self._forward(x)
        m = y.mean(dim=0)
        s = self.init_scale / (y.std(dim=0) + _SMALL)
        assert m.shape == s.shape == self.b.shape
        self.w.copy_(self.w * s[:, None])
        self.b.copy_(-m * s)
        return self._forward(x)

    def _forward(self, x):
        return F.linear(x, self.w, self.b[None, :])


class WnDense(DataDepInitModule):
    def __init__(self, in_features, out_features, init_scale=1.0):
        super().__init__()
        self.in_features, self.out_features, self.init_scale = in_features, out_features, init_scale

        self.v = Parameter(torch.Tensor(out_features, in_features))
        self.g = Parameter(torch.Tensor(out_features))
        self.b = Parameter(torch.Tensor(out_features))

        init.normal_(self.v, 0., _WN_INIT_STDV)
        init.ones_(self.g)
        init.zeros_(self.b)

    def _init(self, x):
        # calculate unnormalized activations
        y_unnormalized = self._forward(x)
        # set g and b so that activations are normalized
        m = y_unnormalized.mean(dim=0)
        s = self.init_scale / (y_unnormalized.std(dim=0) + _SMALL)
        assert m.shape == s.shape == self.g.shape == self.b.shape
        self.g.data.copy_(s)
        self.b.data.sub_(m * s)
        # forward pass again, now normalized
        return self._forward(x)

    def _forward(self, x):
        (bs, in_features), out_features = x.shape, self.v.shape[0]
        assert in_features == self.v.shape[1]
        vnorm = self.v.norm(p=2, dim=1)
        assert vnorm.shape == self.g.shape == self.b.shape
        y = torch.addcmul(self.b[None, :], (self.g / vnorm)[None, :], x @ self.v.t())
        # the line above is equivalent to: y = self.b[None, :] + (self.g / vnorm)[None, :] * (x @ self.v.t())
        assert y.shape == (bs, out_features)
        return y

    def extra_repr(self):
        return f'in_features={self.in_dim}, out_features={self.out_features}, init_scale={self.init_scale}'


class _Nin(DataDepInitModule):
    def __init__(self, in_features, out_features, wn: bool, init_scale: float):
        super().__init__()
        base_module = WnDense if wn else Dense
        self.dense = base_module(in_features=in_features, out_features=out_features, init_scale=init_scale)
        self.height, self.width = None, None

    def _preprocess(self, x):
        """(b,c,h,w) -> (b*h*w,c)"""
        B, C, H, W = x.shape
        if self.height is None or self.width is None:
            self.height, self.width = H, W
        else:
            assert self.height == H and self.width == W, 'nin input image shape changed!'
        assert C == self.dense.in_features
        return x.permute(0, 2, 3, 1).reshape(B * H * W, C)

    def _postprocess(self, x):
        """(b*h*w,c) -> (b,c,h,w)"""
        BHW, C = x.shape
        out = x.reshape(-1, self.height, self.width, C).permute(0, 3, 1, 2)
        assert out.shape[1:] == (self.dense.out_features, self.height, self.width)
        return out

    def _init(self, x):
        return self._postprocess(self.dense._init(self._preprocess(x)))

    def _forward(self, x):
        return self._postprocess(self.dense._forward(self._preprocess(x)))


class Nin(_Nin):
    def __init__(self, in_features, out_features, init_scale=1.0):
        super().__init__(in_features=in_features, out_features=out_features, wn=False, init_scale=init_scale)


class WnNin(_Nin):
    def __init__(self, in_features, out_features, init_scale=1.0):
        super().__init__(in_features=in_features, out_features=out_features, wn=True, init_scale=init_scale)


class Conv2d(DataDepInitModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, init_scale=1.0):
        super().__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.init_scale = \
            in_channels, out_channels, kernel_size, stride, padding, dilation, init_scale

        self.w = Parameter(torch.Tensor(out_channels, in_channels, self.kernel_size, self.kernel_size))
        self.b = Parameter(torch.Tensor(out_channels))

        init.normal_(self.w, 0, _WN_INIT_STDV)
        init.zeros_(self.b)

    def _init(self, x):
        # x.shape == (batch, channels, h, w)
        y = self._forward(x)  # (batch, out_channels, h, w)
        m = y.transpose(0, 1).reshape(y.shape[1], -1).mean(dim=1)  # out_channels
        s = self.init_scale / (y.transpose(0, 1).reshape(y.shape[1], -1).std(dim=1) + _SMALL)  # out_channels
        self.w.copy_(self.w * s[:, None, None, None])  # (out, in, k, k) * (ou))
        self.b.copy_(-m * s)
        return self._forward(x)

    def _forward(self, x):
        return F.conv2d(x, self.w, self.b, self.stride, self.padding, self.dilation, 1)


class WnConv2d(DataDepInitModule):
    def __init__(self, in_channels, out_channels, kernel_size, padding, init_scale=1.0):
        super().__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.padding = in_channels, out_channels, kernel_size, padding
        self.init_scale = init_scale

        self.v = Parameter(torch.Tensor(out_channels, in_channels, self.kernel_size, self.kernel_size))
        self.g = Parameter(torch.Tensor(out_channels))
        self.b = Parameter(torch.Tensor(out_channels))

        init.normal_(self.v, 0., _WN_INIT_STDV)
        init.ones_(self.g)
        init.zeros_(self.b)

    def _init(self, x):
        # calculate unnormalized activations
        y_bchw = self._forward(x)
        assert len(y_bchw.shape) == 4 and y_bchw.shape[:2] == (x.shape[0], self.out_channels)
        # set g and b so that activations are normalized
        y_c = y_bchw.transpose(0, 1).reshape(self.out_channels, -1)
        m = y_c.mean(dim=1)
        s = self.init_scale / (y_c.std(dim=1) + _SMALL)
        assert m.shape == s.shape == self.g.shape == self.b.shape
        self.g.data.copy_(s)
        self.b.data.sub_(m * s)
        # forward pass again, now normalized
        return self._forward(x)

    def _forward(self, x):
        vnorm = self.v.view(self.out_channels, -1).norm(p=2, dim=1)
        assert vnorm.shape == self.g.shape == self.b.shape
        w = self.v * (self.g / (vnorm + _SMALL)).view(self.out_channels, 1, 1, 1)
        return F.conv2d(x, w, self.b, padding=self.padding)

    def extra_repr(self):
        return f'in_channels={self.in_dim}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, padding={self.padding}, init_scale={self.init_scale}'


class LearnedNorm(DataDepInitModule):
    def __init__(self, shape, init_scale=1.0):
        super().__init__()
        self.init_scale = init_scale
        self.g = Parameter(torch.ones(*shape))
        self.b = Parameter(torch.zeros(*shape))

    def _init(self, x, *, inverse):
        assert not inverse
        assert x.shape[1:] == self.g.shape == self.b.shape
        m_init = x.mean(dim=0)
        scale_init = self.init_scale / (x.std(dim=0) + _SMALL)
        self.g.copy_(scale_init)
        self.b.copy_(-m_init * scale_init)
        return self._forward(x, inverse=inverse)

    def get_gain(self):
        return torch.clamp(self.g, min=1e-10)

    def _forward(self, x, *, inverse):
        """
        inverse == False to normalize; inverse == True to unnormalize
        """
        assert x.shape[1:] == self.g.shape == self.b.shape
        assert x.dtype == self.g.dtype == self.b.dtype
        g = self.get_gain()
        if not inverse:
            return x * g[None] + self.b[None]
        else:
            return (x - self.b[None]) / g[None]


@torch.no_grad()
def _test_data_dep_init(m, x, init_scale, verbose=True, tol=1e-8, kwargs=None):
    if kwargs is None:
        kwargs = {}
    with init_mode():
        y_init = m(x, **kwargs)
    y = m(x, **kwargs)
    assert (y - y_init).abs().max() < tol, 'init pass output does not match normal forward pass'
    y_outputs_flat = y.transpose(0, 1).reshape(y.shape[1], -1)  # assumes axis 1 is the output axis
    assert y_outputs_flat.mean(dim=1).abs().max() < tol, 'means wrong after normalization'
    assert (y_outputs_flat.std(dim=1) - init_scale).abs().max() < tol, 'standard deviations wrong after normalization'
    if verbose:
        print('ok')


def test_dense():
    bs = 128
    in_features = 20
    out_features = 29
    init_scale = 3.14159
    x = torch.randn(bs, in_features, dtype=torch.float64)
    for module in [Dense, WnDense]:
        m = module(in_features=in_features, out_features=out_features, init_scale=init_scale).double()
        _test_data_dep_init(m, x, init_scale)
        assert m(x).shape == (bs, out_features)


def test_conv2d():
    bs = 128
    in_channels = 20
    out_channels = 29
    height = 9
    width = 11
    init_scale = 3.14159
    x = torch.randn(bs, in_channels, height, width, dtype=torch.float64)
    for module in [Conv2d, WnConv2d]:
        m = module(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
                   init_scale=init_scale).double()
        _test_data_dep_init(m, x, init_scale)
        assert m(x).shape == (bs, out_channels, height, width)


def test_learnednorm():
    bs = 128
    in_features = 20
    init_scale = 3.14159
    x = torch.rand(bs, in_features, dtype=torch.float64)
    m = LearnedNorm(shape=(in_features,), init_scale=init_scale).double()
    _test_data_dep_init(m, x, init_scale, kwargs={'inverse': False})
    y = m(x, inverse=False)
    assert y.shape == (bs, in_features)
    assert torch.allclose(m(y, inverse=True), x), 'inverse failed'
