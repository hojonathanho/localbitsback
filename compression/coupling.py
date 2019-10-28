import math

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter
import numpy as np

from . import flows
from .flows import BaseFlow, Compose, Inverse, _run_flow_test, _run_compression_test, Bitstream
from .nn import Conv2d, Nin
from .utils import sumflat


class Parallel(BaseFlow):
    def __init__(self, flow_constructors):
        super().__init__()
        assert len(flow_constructors) > 0
        self.flows = ModuleList([ctor() for ctor in flow_constructors])

    def forward(self, input_, *, aux, inverse):
        assert isinstance(input_, tuple) and len(input_) == len(self.flows)
        ys = []
        total_logd = 0.
        for x, f in zip(input_, self.flows):
            y, logd = f(x, aux=aux, inverse=inverse)
            assert y.shape[0] == x.shape[0] and logd.shape == (x.shape[0],)
            ys.append(y)
            total_logd += logd
        assert all(total_logd.shape[0] == input_[0].shape[0] == y.shape[0] for y in ys)
        return tuple(ys), total_logd

    def code(self, input_sym, *, aux, inverse: bool, stream: Bitstream):
        assert isinstance(input_sym, tuple) and len(input_sym) == len(self.flows)
        pairs = list(zip(input_sym, self.flows))
        y_syms = []
        for x_sym, f in (reversed(pairs) if inverse else pairs):
            y_syms.append(f.code(x_sym, aux=aux, inverse=inverse, stream=stream))
        return tuple(reversed(y_syms) if inverse else y_syms)


def test_parallel():
    construct = lambda: Compose([StripeSplit(), Parallel([flows.Sigmoid, flows.ImgProc]), Inverse(StripeSplit())])
    _run_flow_test(construct, x_bounds=(0., 1.))
    _run_compression_test(construct, x_bounds=(0., 1.))


class NoCompressionFlow(BaseFlow):
    """A flow that does nothing to the compression bitstream"""

    @staticmethod
    def _convert(xs, to_torch):
        if not isinstance(xs, tuple):
            input_is_tuple = False
            xs = (xs,)
        else:
            input_is_tuple = True

        if to_torch:
            ys = tuple(torch.from_numpy(s.view(dtype=np.int64)) for s in xs)
        else:
            ys = tuple(s.numpy().view(dtype=np.uint64) for s in xs)

        if not input_is_tuple:
            ys = ys[0]
        return ys

    def code(self, input_sym, *, aux, inverse: bool, stream: Bitstream):
        out, _ = self(self._convert(input_sym, True), aux=aux, inverse=inverse)
        return self._convert(out, False)


class TupleFlip(BaseFlow):
    @staticmethod
    def _flip(t):
        assert isinstance(t, tuple)
        a, b = t
        return b, a

    def forward(self, input_, *, aux, inverse: bool):
        return self._flip(input_), None

    def code(self, input_sym, *, aux, inverse: bool, stream: Bitstream):
        return self._flip(input_sym)


class ChannelSplit(NoCompressionFlow):
    def forward(self, input_, *, aux, inverse: bool):
        if not inverse:
            x_bchw = input_
            assert len(x_bchw.shape) == 4 and x_bchw.shape[1] % 2 == 0
            return torch.chunk(x_bchw, 2, dim=1), None
        else:
            assert isinstance(input_, tuple)
            a, b = input_
            return torch.cat((a, b), dim=1), None


def test_channel_split():
    construct = lambda: Compose([ChannelSplit(), Inverse(ChannelSplit())])
    _run_flow_test(construct, x_shape=(4, 8, 8))
    _run_compression_test(construct, x_shape=(4, 8, 8))


class StripeSplit(NoCompressionFlow):
    def forward(self, input_, *, aux, inverse: bool):
        if not inverse:
            x_bchw = input_
            B, C, H, W = x_bchw.shape
            return x_bchw.reshape(B, C, H, W // 2, 2).unbind(dim=4), None
        else:
            assert isinstance(input_, tuple)
            a, b = input_
            assert a.shape == b.shape
            B, C, H, W_half = a.shape
            return torch.stack((a, b), dim=4).reshape(B, C, H, W_half * 2), None


def test_stripe_split():
    construct = lambda: Compose([StripeSplit(), Inverse(StripeSplit())])
    _run_flow_test(construct)
    _run_compression_test(construct)


def space_to_depth(x):
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0
    # return x.reshape(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, H // 2, W // 2)
    # below: compatible with tf.space_to_depth
    return x.reshape(B, C, H // 2, 2, W // 2, 2).permute(0, 3, 5, 1, 2, 4).reshape(B, C * 4, H // 2, W // 2)


def depth_to_space(y):
    B, C_4, H_half, W_half = y.shape
    assert C_4 % 4 == 0
    C, H, W = C_4 // 4, H_half * 2, W_half * 2
    # return y.reshape(B, C, 2, 2, H_half, W_half).permute(0, 1, 4, 2, 5, 3).reshape(B, C, H, W)
    return y.reshape(B, 2, 2, C, H_half, W_half).permute(0, 3, 4, 1, 5, 2).reshape(B, C, H, W)


@torch.no_grad()
def test_space_to_depth():
    bs = 3
    ch = 4
    height = 8
    width = 8
    shape = [bs, ch, height, width]
    x = torch.randn(*shape)
    y = space_to_depth(x)

    y2 = torch.stack((
        x[:, :, ::2, ::2],
        x[:, :, ::2, 1::2],
        x[:, :, 1::2, ::2],
        x[:, :, 1::2, 1::2]
    ), dim=1).reshape(bs, ch * 4, height // 2, width // 2)
    assert torch.allclose(y, y2)

    x2 = depth_to_space(y)
    assert torch.allclose(x, x2)


class Squeeze(NoCompressionFlow):
    def forward(self, input_, *, aux, inverse: bool):
        return (depth_to_space if inverse else space_to_depth)(input_), None


def test_squeeze():
    construct = lambda: Compose([Squeeze(), Inverse(Squeeze())])
    _run_flow_test(construct)
    _run_compression_test(construct)


##################################################


def concat_elu(x, *, dim=1):
    return F.elu(torch.cat((x, -x), dim=dim))


def _gate(x, *, dim=1):
    a, b = torch.chunk(x, 2, dim=dim)
    return a * torch.sigmoid(b)


class ConvGate(Module):
    def __init__(self, *, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels * 2,
                           kernel_size=kernel_size, padding=padding, init_scale=0.1)

    def forward(self, x):
        assert len(x.shape) == 4
        return _gate(self.conv(x))


class NinGate(Module):
    def __init__(self, *, in_features, out_features):
        super().__init__()
        self.nin = Nin(in_features=in_features, out_features=out_features * 2, init_scale=0.1)

    def forward(self, x):
        return _gate(self.nin(x))


class GatedConv(Module):
    def __init__(self, *, in_channels: int, aux_channels: int, gate_nin=False, pdrop: float):
        super().__init__()
        assert isinstance(aux_channels, int)

        self.conv = Conv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.aux_proj = Nin(in_features=2 * aux_channels, out_features=in_channels) if aux_channels > 0 else None
        self.dropout = torch.nn.Dropout(pdrop)
        if gate_nin:
            self.gate = NinGate(in_features=2 * in_channels, out_features=in_channels)
        else:
            self.gate = ConvGate(in_channels=2 * in_channels, out_channels=in_channels)

    def forward(self, x, *, aux=None):
        h = self.conv(concat_elu(x))
        assert (aux is None) == (self.aux_proj is None)
        if self.aux_proj is not None:
            assert aux.shape[0] == x.shape[0]
            h += self.aux_proj(concat_elu(aux))
        h = concat_elu(h)
        h = self.dropout(h)
        h = self.gate(h)
        return x + h


class GatedAttention(Module):
    def __init__(self, *, in_channels: int, heads: int, pdrop: float):
        super().__init__()
        assert in_channels % heads == 0
        self.in_channels, self.heads, self.head_features = in_channels, heads, in_channels // heads
        self.proj_in = Nin(in_features=in_channels, out_features=3 * in_channels)
        self.gate = NinGate(in_features=in_channels, out_features=in_channels)
        self.dropout = torch.nn.Dropout(pdrop)

    def forward(self, x, *, pos_emb):
        bs, in_channels, height, width = x.shape
        assert in_channels == self.in_channels
        timesteps = height * width

        # Add in position embeddings and project up
        assert pos_emb.shape == x.shape[1:]
        c = x + pos_emb[None, ...]
        # assert pos_emb.shape == x.shape
        # c = x + pos_emb  # don't broadcast over dim 0 because we need this to work with DataParallel
        c = self.proj_in(c)
        assert c.shape == (bs, 3 * in_channels, height, width)

        # Split into Q/K/V
        c = c.reshape(bs, 3, self.heads, self.head_features, timesteps)
        c = c.permute(1, 0, 2, 4, 3)
        assert c.shape == (3, bs, self.heads, timesteps, self.head_features)
        q_bhtd, k_bhtd, v_bhtd = c.unbind(dim=0)
        assert q_bhtd.shape == k_bhtd.shape == v_bhtd.shape == (bs, self.heads, timesteps, self.head_features)

        # Attention
        w_bhtt = torch.matmul(q_bhtd, k_bhtd.transpose(2, 3)) / math.sqrt(float(self.head_features))
        assert w_bhtt.shape == (bs, self.heads, timesteps, timesteps)
        w_bhtt = F.softmax(w_bhtt, dim=3)
        a_bhtd = torch.matmul(w_bhtt, v_bhtd)
        assert a_bhtd.shape == (bs, self.heads, timesteps, self.head_features)

        # Reshape to image and project out
        a = a_bhtd.permute(0, 1, 3, 2)  # b, h, d, t
        assert a.shape == (bs, self.heads, self.head_features, timesteps)
        a = a.reshape(bs, self.heads * self.head_features, height, width)  # b, c, h, w
        assert a.shape == x.shape
        a = self.dropout(a)
        a = self.gate(a)
        return x + a


class ImgLayerNorm(Module):
    """
    LayerNorm for images with channel axis 1
    (this is necessary because PyTorch's LayerNorm operates on the last axis)
    """

    def __init__(self, in_dim, eps=1e-5):
        super().__init__()
        self.in_dim = in_dim
        self.layernorm = torch.nn.LayerNorm(in_dim, eps=eps)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_dim
        # Move the channel axis to the end, apply layernorm, then move it back
        out = self.layernorm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        assert out.shape == x.shape
        return out


@torch.no_grad()
def test_img_layernorm():
    x = torch.rand(3, 10, 4, 4)
    m = ImgLayerNorm(x.shape[1], eps=1e-10)
    y = m(x)
    assert y.shape == x.shape
    assert y.mean(dim=1).abs().max() < 1e-6
    assert (y.std(dim=1, unbiased=False) - 1).abs().max() < 1e-6


class ConvAttnBlock(Module):
    def __init__(self, *, channels, aux_channels, attn_heads, pdrop, attn_version):
        super().__init__()
        self.conv = GatedConv(in_channels=channels, aux_channels=aux_channels, gate_nin=True, pdrop=pdrop)
        self.ln1 = ImgLayerNorm(channels)
        if attn_version:
            self.attn = GatedAttention(in_channels=channels, heads=attn_heads, pdrop=pdrop)
            self.ln2 = ImgLayerNorm(channels)
        self.attn_version = attn_version

    def forward(self, x, *, aux, pos_emb):
        B, C, H, W = x.shape
        assert pos_emb.shape == (C, H, W)
        # pos_emb = pos_emb.unsqueeze(0).repeat(B, 1, 1, 1)
        x = self.conv(x, aux=aux)
        x = self.ln1(x)
        if self.attn_version:
            x = self.attn(x, pos_emb=pos_emb)
            x = self.ln2(x)
        return x


class ConvAttnStack(Module):
    def __init__(self, *, img_shape, hidden_channels, out_channels, aux_channels,
                 blocks, attn_heads, pdrop, output_init_scale, attn_version,
                 pos_emb_init=0.01):  # TODO this number
        super().__init__()

        in_channels, height, width = img_shape

        self.pos_emb = Parameter(torch.Tensor(hidden_channels, height, width))
        torch.nn.init.normal_(self.pos_emb, mean=0., std=pos_emb_init)

        self.proj_in = Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.blocks = ModuleList([
            ConvAttnBlock(channels=hidden_channels, aux_channels=aux_channels, attn_heads=attn_heads, pdrop=pdrop, attn_version=attn_version)
            for _ in range(blocks)
        ])
        self.proj_out = Conv2d(in_channels=hidden_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, init_scale=output_init_scale)

    def forward(self, x, *, aux):
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x, aux=aux, pos_emb=self.pos_emb)
        x = self.proj_out(x)
        return x


class MixLogisticConvAttnCoupling(BaseFlow):
    def __init__(self, *, cf_shape, hidden_channels: int, aux_channels: int,
                 blocks: int, mix_components: int, attn_heads: int, pdrop: float,
                 force_float32_cond: bool, attn_version=True):
        # force_float32_cond: if True, then the conditioning network is always run in float32, even if the data being
        # flowed through the model is in float64.
        super().__init__()
        in_channels, _, _ = cf_shape
        self.mix_components = mix_components
        self.num_trans_params = 2 + 3 * mix_components  # 2 for affine, 3*mix_components for logistic mixture
        self.cond = ConvAttnStack(
            img_shape=cf_shape, hidden_channels=hidden_channels, out_channels=in_channels * self.num_trans_params,
            aux_channels=aux_channels, blocks=blocks, attn_heads=attn_heads, pdrop=pdrop, output_init_scale=0.1,
            attn_version=attn_version
        )
        self.force_float32_cond = force_float32_cond

    def _make_elementwise_flow(self, *, cf, aux):
        # Compute elementwise transformation parameters, conditioned on `cf`
        bs, chns, cf_h, cf_w = cf.shape
        if self.force_float32_cond:
            assert cf.dtype == torch.float64, 'This flag is only meaningful when the rest of the network is in float64'
            self.cond.float()
            tparams = self.cond(cf.float(), aux=None if aux is None else aux.float()).to(cf.dtype)
        else:
            tparams = self.cond(cf, aux=aux)

        assert tparams.shape == (bs, chns * self.num_trans_params, cf_h, cf_w)
        # Extract the elementwise transformation parameters (here: axis ordering for TF checkpoint compatibility)
        tparams = tparams.reshape(bs, chns, self.num_trans_params, cf_h, cf_w).permute(0, 2, 1, 3, 4)
        aff_logscale = torch.tanh(tparams[:, 0, :, :, :])  # affine log-scale
        aff_translation = tparams[:, 1, :, :, :]  # affine translation
        logits, means, logscales = torch.chunk(tparams[:, 2:, :, :, :], 3, dim=1)  # mixture of logistics parameters
        assert aff_logscale.shape == aff_translation.shape == cf.shape
        assert logits.shape == means.shape == logscales.shape == (bs, self.mix_components, *cf.shape[1:])
        # Make the flow
        return Compose([
            flows.MixLogisticCDF(logits=logits, means=means, logscales=logscales, mix_dim=1),
            flows.Inverse(flows.Sigmoid()),
            flows.ElementwiseAffine(logscales=aff_logscale, translations=aff_translation)
        ])

    def forward(self, input_, *, aux, inverse: bool):
        assert isinstance(input_, tuple)
        cf, ef = input_
        ef_flow = self._make_elementwise_flow(cf=cf, aux=aux)
        out_ef, logd = ef_flow(ef, aux=None, inverse=inverse)
        return (cf, out_ef), sumflat(logd)

    def code(self, input_sym, *, aux, inverse: bool, stream: Bitstream):
        assert isinstance(input_sym, tuple)
        cf_sym, ef_sym = input_sym
        ef_flow = self._make_elementwise_flow(cf=stream.from_sym(cf_sym), aux=aux)
        out_ef_sym = ef_flow.code(ef_sym, aux=None, inverse=inverse, stream=stream)
        return cf_sym, out_ef_sym


@torch.no_grad()
def test_coupling():
    x_shape = (3, 4, 4)

    def construct():
        return Compose([
            StripeSplit(),
            MixLogisticConvAttnCoupling(
                cf_shape=(3, 4, 2),
                hidden_channels=16,
                aux_channels=x_shape[0],
                blocks=2,
                mix_components=7,
                attn_heads=4,
                pdrop=0
            ),
            Inverse(StripeSplit()),
        ])

    _run_flow_test(construct, x_shape=x_shape, aux_shape=(3, 4, 2), finitediff_eps=1e-5)
    _run_compression_test(construct, x_shape=x_shape, aux_shape=(3, 4, 2), bs=8)
