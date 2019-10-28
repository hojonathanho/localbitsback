from contextlib import contextmanager

import numpy as np
import torch
import torch.nn
from fast_ans import Discretization, ANS


def is_sym_type(x):
    """Check if x is an array of symbols suitable for ANS"""
    return isinstance(x, np.ndarray) and x.dtype == np.uint64


def is_real_type(x):
    return isinstance(x, torch.Tensor) and x.dtype in (torch.float32, torch.float64)


class Bitstream:
    def __init__(self, *,
                 device,
                 load_from_dict=None,
                 noise_scale: float = None, disc_bits: int = None, disc_range: int = None,
                 ans_mass_bits: int = None, ans_init_bits: int = None,
                 ans_init_seed: int = None, ans_num_streams: int = None):
        self._device = device

        self._pad_encode = False
        self._pad_decode = False

        if load_from_dict:
            self.from_dict(load_from_dict)
        else:
            self._ans = ANS(
                ans_mass_bits=ans_mass_bits, ans_init_bits=ans_init_bits, ans_init_seed=ans_init_seed,
                num_streams=ans_num_streams
            )
            self._disc = Discretization(lo=-disc_range, hi=disc_range, bits=disc_bits)
            self._disc_bits, self._disc_range = disc_bits, disc_range  # TODO expose these in Discretization?
            self._default_noise_scale = noise_scale

        self._decode_monitor_list = None

    def to_dict(self):
        return {
            'ans': self._ans.to_py(),
            'noise_scale': self._default_noise_scale,
            'disc_bits': self._disc_bits,
            'disc_range': self._disc_range,
        }

    def from_dict(self, data):
        self._disc = Discretization(lo=-data['disc_range'], hi=data['disc_range'], bits=data['disc_bits'])
        self._ans = ANS(data['ans'])
        self._default_noise_scale = data['noise_scale']

    def __len__(self):
        return self._ans.stream_length()

    def from_sym(self, sym):
        assert is_sym_type(sym)
        return torch.from_numpy(self._disc.symbol_to_real(sym)).to(self._device)

    def to_sym(self, x):
        assert is_real_type(x)
        return self._disc.real_to_symbol(x.cpu().numpy())

    @contextmanager
    def set_padding(self, encode=False, decode=False):
        old_vals = (self._pad_encode, self._pad_decode)
        self._pad_encode, self._pad_decode = encode, decode
        yield
        self._pad_encode, self._pad_decode = old_vals

    @contextmanager
    def monitor_decode(self, lst):
        self._decode_monitor_list = lst
        yield
        self._decode_monitor_list = None

    def encode_gaussian_diag(self, x_sym, means, stds, scale=None):
        if isinstance(stds, float): stds = torch.full_like(means, stds)
        assert is_sym_type(x_sym) and is_real_type(means) and is_real_type(stds)
        assert x_sym.shape == means.shape == stds.shape
        if scale is None: scale = self._default_noise_scale
        before = len(self)
        self._ans.encode_gaussian_diag(
            x_sym.ravel(),
            means=means.cpu().numpy().ravel(),
            stds=(scale * stds).cpu().numpy().ravel(),
            disc=self._disc,
            pad=self._pad_encode
        )

    def decode_gaussian_diag(self, means, stds, scale=None):
        if isinstance(stds, float): stds = torch.full_like(means, stds)
        assert is_real_type(means) and is_real_type(stds)
        assert means.shape == stds.shape
        if scale is None: scale = self._default_noise_scale
        before = len(self)
        out = self._ans.decode_gaussian_diag(
            means=means.cpu().numpy().ravel(),
            stds=(scale * stds).cpu().numpy().ravel(),
            disc=self._disc,
            pad=self._pad_decode
        ).reshape(means.shape)
        if self._decode_monitor_list is not None:
            self._decode_monitor_list.append(len(self))
        return out

    def encode_gaussian(self, x_syms, mean_coefs, biases, stds, scale=None):
        assert is_sym_type(x_syms)
        bs, dim = x_syms.shape
        assert x_syms.shape == biases.shape
        assert mean_coefs.shape == (dim, dim) and stds.shape == (dim,)
        if scale is None: scale = self._default_noise_scale
        before = len(self)
        self._ans.encode_gaussian_batched(
            x_syms,
            mean_coefs=mean_coefs.cpu().numpy(),
            biases=biases.cpu().numpy(),
            stds=(scale * stds).cpu().numpy(),
            left_to_right=True,
            disc=self._disc,
            pad=self._pad_encode
        )

    def decode_gaussian(self, mean_coefs, biases, stds, scale=None):
        if scale is None: scale = self._default_noise_scale
        before = len(self)
        out = self._ans.decode_gaussian_batched(
            mean_coefs=mean_coefs.cpu().numpy(),
            biases=biases.cpu().numpy(),
            stds=(scale * stds).cpu().numpy(),
            left_to_right=True,
            disc=self._disc,
            pad=self._pad_decode
        )
        if self._decode_monitor_list is not None:
            self._decode_monitor_list.append(len(self))
        return out


class CompressionModel(torch.nn.Module):
    def __init__(self, main_flow, dequant_flow, x_shape, z_shape):
        super().__init__()
        self.main_flow = main_flow
        self.dequant_flow = dequant_flow
        assert isinstance(x_shape, tuple) and isinstance(z_shape, tuple)
        self.x_shape = x_shape
        self.z_shape = z_shape

    def encode(self, x_raw: torch.Tensor, *, stream: Bitstream):
        assert x_raw.dtype == torch.int64 and x_raw.shape[1:] == self.x_shape
        x_real = x_raw.to(dtype=torch.float64)
        x_sym = stream.to_sym(x_real)
        assert (stream.from_sym(x_sym).long() == x_raw).all(), \
            'converting raw data to symbols should never lose information'

        post_decode_lengths = []
        # start_len = len(stream)
        with stream.set_padding(encode=True), stream.monitor_decode(post_decode_lengths):
            eps_sym = stream.decode_gaussian_diag(
                means=torch.zeros(x_raw.shape, dtype=torch.float64), stds=1.0, scale=1.0
            )
            u_sym = self.dequant_flow.code(eps_sym, aux=x_real, inverse=False, stream=stream)
            assert x_sym.shape == u_sym.shape and x_sym.dtype == u_sym.dtype == np.uint64

            # dequantizing: OK to add symbols because of how discretization is defined
            # NOTE do not need subtraction if we use different discretization bounds for u
            if ((not (u_sym >= stream.to_sym(torch.zeros(u_sym.shape))).all()) or
                    (not (u_sym <= stream.to_sym(torch.ones(u_sym.shape))).all())):
                print('WARNING: dequant noise not in [0, 1] bounds!')
            dequantized_x_sym = x_sym + (u_sym - stream.to_sym(torch.zeros(u_sym.shape)))

            z_sym = self.main_flow.code(dequantized_x_sym, aux=None, inverse=False, stream=stream)
            assert z_sym.shape[1:] == self.z_shape

            stream.encode_gaussian_diag(z_sym, means=torch.zeros(z_sym.shape, dtype=torch.float64), stds=1.0, scale=1.0)

        return dict(  # debugging info
            x_sym=x_sym, eps_sym=eps_sym, u_sym=u_sym,
            dequantized_x_sym=dequantized_x_sym, z_sym=z_sym,
            post_decode_lengths=post_decode_lengths
        )

    def decode(self, *, bs: int, stream: Bitstream, encoding_dbg_info=None):
        with stream.set_padding(decode=True):
            # decode z
            z_sym = stream.decode_gaussian_diag(
                means=torch.zeros((bs, *self.z_shape), dtype=torch.float64), stds=1.0, scale=1.0
            )
            if encoding_dbg_info is not None:
                assert (z_sym == encoding_dbg_info['z_sym']).all()

            # decode x+u | z
            dequantized_x_sym = self.main_flow.code(z_sym, aux=None, inverse=True, stream=stream)
            if encoding_dbg_info is not None:
                assert (dequantized_x_sym == encoding_dbg_info['dequantized_x_sym']).all()

            # round down to obtain x from x+u (dequantized x)
            rounded_x = stream.from_sym(dequantized_x_sym).floor()
            x_sym = stream.to_sym(rounded_x)
            if encoding_dbg_info is not None:
                assert (x_sym == encoding_dbg_info['x_sym']).all()
            u_sym = dequantized_x_sym - x_sym + stream.to_sym(torch.zeros(x_sym.shape))  # NOTE again the shift
            if encoding_dbg_info is not None:
                assert (u_sym == encoding_dbg_info['u_sym']).all()

            # decode eps | u (given x)
            eps_sym = self.dequant_flow.code(u_sym, aux=rounded_x, inverse=True, stream=stream)
            if encoding_dbg_info is not None:
                assert (eps_sym == encoding_dbg_info['eps_sym']).all()

            # encode eps
            stream.encode_gaussian_diag(
                eps_sym,
                means=torch.zeros(eps_sym.shape, dtype=torch.float64), stds=1.0, scale=1.0
            )
        x_raw = rounded_x.to(dtype=torch.int64)
        assert x_raw.shape == (bs, *self.x_shape)
        return x_raw
