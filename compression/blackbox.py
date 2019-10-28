"""
Local bits-back coding, black box algorithm
Calculates the Jacobian of the flow, makes no use of the structure of the flow
"""

from contextlib import contextmanager

import fast_ans
import numpy as np
import torch
from tqdm import trange

from compression.utils import process_gaussian


@torch.enable_grad()
def compute_jacobian(fn, x0: torch.Tensor, bs: int):
    """
    Computes the Jacobian matrix of the given function at x0, using vector-Jacobian products
    """
    input_shape = x0.shape
    assert len(input_shape) == 3
    dim = x0.numel()
    eye = torch.eye(dim, dtype=x0.dtype, device=x0.device)

    # Forward pass
    x0rep = x0.detach()[None].repeat([bs] + [1] * len(input_shape))  # repeat along batch axis
    x0rep.requires_grad = True
    z0rep = fn(x0rep)
    zshape = z0rep.shape[1:]
    assert zshape.numel() == dim

    # Compute batches of rows of the Jacobian
    rows = []
    for row_start in trange(0, dim, bs, desc='jacobian', leave=False):
        # Pre-pad with extra rows to ensure that batch size stays constant
        row_end = min(row_start + bs, dim)
        num_rows = row_end - row_start
        if num_rows != bs:
            assert num_rows < bs
            pre_pad_rows = bs - num_rows
        else:
            pre_pad_rows = 0
        assert row_start - pre_pad_rows >= 0
        # vector-Jacobian product with rows of an identity matrix
        g, = torch.autograd.grad(
            z0rep, x0rep,
            grad_outputs=eye[row_start - pre_pad_rows:row_end].reshape(row_end - row_start + pre_pad_rows, *zshape),
            retain_graph=True
        )
        assert g.shape == x0rep.shape
        rows.append(g.view(g.shape[0], -1)[pre_pad_rows:, :])

    jacobian = torch.cat(rows, dim=0)
    assert jacobian.shape == (dim, dim)
    return jacobian


class BlackBoxBitstream:
    def __init__(self, *, model_ctor, model_filename, device, jacobian_bs,
                 disc_bits=32, xnoise_std=2 ** (-20), ans_init_bits=int(1e7)):
        self.device = device
        self.xnoise_std = xnoise_std
        self.ans = fast_ans.ANS(ans_mass_bits=60, ans_init_bits=ans_init_bits, ans_init_seed=0, num_streams=16)
        self.disc_bits = disc_bits
        self.init_bitstream_length = self.ans.stream_length()
        # discretization for data and latents
        self.disc = fast_ans.Discretization(lo=-256, hi=256, bits=disc_bits)
        # discretization for dequantization noise
        self.u_disc = fast_ans.Discretization(lo=0, hi=1, bits=disc_bits)

        self.jacobian_bs = jacobian_bs  # batch size for computing the Jacobian matrix
        self.model64 = model_ctor(model_filename, force_float32_cond=False).to(device=device, dtype=torch.float64)
        self.model32 = model_ctor(model_filename, force_float32_cond=False).to(device=device, dtype=torch.float32)
        self.x_shape = self.model64.x_shape
        self.z_shape = self.model64.z_shape
        self.dim = int(np.prod(self.x_shape))
        assert int(np.prod(self.z_shape)) == self.dim

    def __len__(self):
        return self.ans.stream_length()

    def _sym_to_torch(self, a, disc):
        assert isinstance(a, np.ndarray) and a.dtype == np.uint64
        a = torch.from_numpy(disc.symbol_to_real(a)).to(self.device)
        assert a.dtype == torch.float64
        return a

    @contextmanager
    def _monitor(self, description):
        """Prints debugging information"""
        bits_before = self.ans.stream_length()
        yield
        bpd = (self.ans.stream_length() - bits_before) / self.dim
        # print(f'{description}: {bpd:.5f} bpd')

    def _encode_generic(self, x_sym, flow_fn, inverse_flow_fn, jacobian_fn, inverse_jacobian_fn,
                        x_shape, x_disc, z_disc, in_name='x', out_name='z'):
        x = self._sym_to_torch(x_sym, x_disc).reshape(x_shape)
        z, = flow_fn(x[None])

        # decode z|x
        with self._monitor(f'decode {out_name}|{in_name}'):
            assert (jacobian_fn is None) != (inverse_jacobian_fn is None)
            if jacobian_fn is not None:
                mean_coefs, stds = process_gaussian(jacobian_fn(x, z), scale=self.xnoise_std, inverted=False)
                znoisy_sym = self.ans.decode_gaussian(
                    mean_coefs=mean_coefs.cpu().numpy(), biases=z.cpu().numpy().ravel(), stds=stds.cpu().numpy(),
                    left_to_right=True, disc=z_disc, pad=True
                )
            else:
                assert inverse_jacobian_fn is not None
                mean_coefs, stds = process_gaussian(inverse_jacobian_fn(x, z), scale=1.0 / self.xnoise_std,
                                                    inverted=True)
                znoisy_sym = self.ans.decode_gaussian(
                    mean_coefs=mean_coefs.cpu().numpy(), biases=z.cpu().numpy().ravel(), stds=stds.cpu().numpy(),
                    left_to_right=False, disc=z_disc, pad=True
                )

        # encode x|z
        with self._monitor(f'encode {in_name}|{out_name}'):
            xnoisy, = inverse_flow_fn(self._sym_to_torch(znoisy_sym, z_disc).reshape(1, *z.shape))
            self.ans.encode_gaussian_diag(
                x_sym, means=xnoisy.cpu().numpy().ravel(), stds=self.xnoise_std * np.ones(self.dim),
                disc=x_disc, pad=True
            )

        # encode z (N(0,I) prior)
        with self._monitor(f'encode {out_name}'):
            self.ans.encode_gaussian_diag(
                znoisy_sym, means=np.zeros(self.dim), stds=np.ones(self.dim), disc=z_disc, pad=True
            )

    def _decode_generic(self, flow_fn, inverse_flow_fn, jacobian_fn, inverse_jacobian_fn,
                        z_shape, x_disc, z_disc, in_name='x', out_name='z'):
        # decode z
        with self._monitor(f'decode {out_name}'):
            znoisy_sym = self.ans.decode_gaussian_diag(
                means=np.zeros(self.dim), stds=np.ones(self.dim), disc=z_disc, pad=True
            )

        # decode x|z
        with self._monitor(f'decode {in_name}|{out_name}'):
            xnoisy, = inverse_flow_fn(self._sym_to_torch(znoisy_sym, z_disc).reshape(1, *z_shape))
            x_sym = self.ans.decode_gaussian_diag(
                means=xnoisy.cpu().numpy().ravel(), stds=self.xnoise_std * np.ones(self.dim), disc=x_disc, pad=True
            )

        # encode z|x
        x = self._sym_to_torch(x_sym, x_disc).reshape(xnoisy.shape)
        z, = flow_fn(x[None])
        with self._monitor(f'encode {out_name}|{in_name}'):
            assert (jacobian_fn is None) != (inverse_jacobian_fn is None)
            if jacobian_fn is not None:
                mean_coefs, stds = process_gaussian(jacobian_fn(x, z), scale=self.xnoise_std, inverted=False)
                self.ans.encode_gaussian(
                    znoisy_sym,
                    mean_coefs=mean_coefs.cpu().numpy(), biases=z.cpu().numpy().ravel(), stds=stds.cpu().numpy(),
                    left_to_right=True, disc=z_disc, pad=True
                )
            else:
                assert inverse_jacobian_fn is not None
                mean_coefs, stds = process_gaussian(inverse_jacobian_fn(x, z), scale=1.0 / self.xnoise_std,
                                                    inverted=True)
                self.ans.encode_gaussian(
                    znoisy_sym,
                    mean_coefs=mean_coefs.cpu().numpy(), biases=z.cpu().numpy().ravel(), stds=stds.cpu().numpy(),
                    left_to_right=False, disc=z_disc, pad=True
                )
        return x_sym

    def _encdec_u(self, u_sym, x_raw):
        assert x_raw.dtype == torch.int64 and x_raw.shape == self.x_shape, 'expected raw integer input in [0, 255]'
        x = x_raw[None].double()  # conditioning info for the dequantizer. batch size 1
        # Note that the forward direction (u -> epsilon) of this flow is actually the inverse of the dequantizer flow
        # we could use the inverse flow directly, but it's more difficult because it involves mixture CDF inversion
        kwargs = dict(
            flow_fn=lambda u_: self.model64.dequant_flow(u_, aux=x, inverse=True)[0],  # inverse flipped here
            inverse_flow_fn=lambda eps_: self.model64.dequant_flow(eps_, aux=x, inverse=False)[0],
            jacobian_fn=None,
            inverse_jacobian_fn=lambda _unused_u_, eps_: compute_jacobian(  # again, we provide the inverse jacobian
                lambda eps__: self.model32.dequant_flow(eps__.float(),
                                                        aux=x.float().repeat(eps__.shape[0], 1, 1, 1),
                                                        inverse=False)[0],
                eps_.float(), bs=self.jacobian_bs
            ).double(),
            x_disc=self.u_disc,
            z_disc=self.disc,
            in_name='u', out_name='e',
        )
        if u_sym is not None:
            assert isinstance(u_sym, np.ndarray) and u_sym.dtype == np.uint64
            self._encode_generic(u_sym, x_shape=self.x_shape, **kwargs)
        else:
            u_sym = self._decode_generic(z_shape=self.x_shape, **kwargs)
            assert u_sym.shape == (x_raw.numel(),)
            return u_sym

    def _decode_u(self, x_raw):
        return self._encdec_u(u_sym=None, x_raw=x_raw)

    def _encode_u(self, u_sym, x_raw):
        return self._encdec_u(u_sym=u_sym, x_raw=x_raw)

    def _encdec_x(self, x_sym):
        kwargs = dict(
            flow_fn=lambda x_: self.model64.main_flow(x_, aux=None, inverse=False)[0],
            inverse_flow_fn=lambda z_: self.model64.main_flow(z_, aux=None, inverse=True)[0],
            jacobian_fn=lambda x_, _unused_z_: compute_jacobian(
                lambda x__: self.model32.main_flow(x__.float(), aux=None, inverse=False)[0],
                x_.float(), bs=self.jacobian_bs
            ).double(),
            inverse_jacobian_fn=None,
            x_disc=self.disc,
            z_disc=self.disc,
            in_name='x', out_name='z',
        )
        if x_sym is not None:
            assert isinstance(x_sym, np.ndarray) and x_sym.dtype == np.uint64
            self._encode_generic(x_sym, x_shape=self.x_shape, **kwargs)
        else:
            return self._decode_generic(z_shape=self.z_shape, **kwargs)

    def _decode_x(self):
        return self._encdec_x(None)

    def _encode_x(self, x_sym):
        return self._encdec_x(x_sym)

    def encode(self, x_raw):
        assert x_raw.dtype == torch.int64 and x_raw.shape == self.x_shape
        x_sym = self.disc.real_to_symbol(x_raw.cpu().numpy()).ravel()
        assert (torch.from_numpy(self.disc.symbol_to_real(x_sym)).long() == x_raw.view(-1).cpu()).all(), \
            'converting raw data to symbols should never lose information'
        u_sym = self._decode_u(x_raw)
        assert x_sym.shape == u_sym.shape == (self.dim,) and x_sym.dtype == u_sym.dtype == np.uint64
        dequantized_x_sym = x_sym + u_sym
        self._encode_x(dequantized_x_sym)

    def decode(self):
        dequantized_x_sym = self._decode_x()
        # round down to obtain x from dequantized x
        rounded_x = np.floor(self.disc.symbol_to_real(dequantized_x_sym))
        x_raw = torch.from_numpy(rounded_x).long().reshape(self.x_shape).to(device=self.device)
        u_sym = dequantized_x_sym - self.disc.real_to_symbol(rounded_x)
        self._encode_u(u_sym, x_raw)
        return x_raw
