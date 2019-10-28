import math

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter

from compression.bitstream import Bitstream
from compression.coupling import ImgLayerNorm, concat_elu
from compression.flows import BaseFlow, Compose, Inverse, Sigmoid, MixLogisticCDF, ElementwiseAffine
from compression.nn import WnNin, WnConv2d
from compression.utils import sumflat


def _gate(x, *, dim=1):
    a, b = torch.chunk(x, 2, dim=dim)
    return a * torch.sigmoid(b)


class NinGate_Imagenet32(Module):
    def __init__(self, *, in_features, out_features):
        super().__init__()
        self.nin = WnNin(in_features=in_features, out_features=out_features * 2, init_scale=0.1)

    def forward(self, x):
        return _gate(self.nin(x))


class ConvGate_Imagenet32(Module):
    def __init__(self, *, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = WnConv2d(in_channels=in_channels, out_channels=out_channels * 2,
                             kernel_size=kernel_size, padding=padding, init_scale=0.1)

    def forward(self, x):
        assert len(x.shape) == 4
        return _gate(self.conv(x))


class GatedAttention_Imagenet32(Module):
    def __init__(self, *, in_channels: int, heads: int, pdrop: float):
        super().__init__()
        assert in_channels % heads == 0
        self.in_channels, self.heads, self.head_features = in_channels, heads, in_channels // heads
        self.proj_in = WnNin(in_features=in_channels, out_features=3 * in_channels)
        self.gate = NinGate_Imagenet32(in_features=in_channels, out_features=in_channels)
        self.dropout = torch.nn.Dropout(pdrop)

    def forward(self, x, *, pos_emb):
        bs, in_channels, height, width = x.shape
        assert in_channels == self.in_channels
        timesteps = height * width

        # Add in position embeddings and project up
        assert pos_emb.shape == x.shape[1:], '{} != {}[1:]'.format(pos_emb.shape, x.shape)
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


class GatedConv_Imagenet32(Module):
    def __init__(self, *, in_channels: int, aux_channels: int, gate_nin=False, pdrop: float):
        super().__init__()
        assert isinstance(aux_channels, int)

        self.conv = WnConv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.aux_proj = WnNin(in_features=2 * aux_channels, out_features=in_channels) if aux_channels > 0 else None
        self.dropout = torch.nn.Dropout(pdrop)
        if gate_nin:
            self.gate = NinGate_Imagenet32(in_features=2 * in_channels, out_features=in_channels)
        else:
            self.gate = ConvGate_Imagenet32(in_channels=2 * in_channels, out_channels=in_channels)

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


class ConvAttnBlock_Imagenet32(Module):
    def __init__(self, *, channels, aux_channels, attn_heads, pdrop):
        super().__init__()
        self.conv = GatedConv_Imagenet32(in_channels=channels, aux_channels=aux_channels, gate_nin=True, pdrop=pdrop)
        self.ln1 = ImgLayerNorm(channels)
        self.attn = GatedAttention_Imagenet32(in_channels=channels, heads=attn_heads, pdrop=pdrop)
        self.ln2 = ImgLayerNorm(channels)

    def forward(self, x, *, aux, pos_emb):
        B, C, H, W = x.shape
        assert pos_emb.shape == (C, H, W)
        # pos_emb = pos_emb.unsqueeze(0).repeat(B, 1, 1, 1)
        x = self.conv(x, aux=aux)
        x = self.ln1(x)
        x = self.attn(x, pos_emb=pos_emb)
        x = self.ln2(x)
        return x


class ConvAttnStack_Imagenet32(Module):
    def __init__(self, *, img_shape, hidden_channels, out_channels, aux_channels,
                 blocks, attn_heads, pdrop, output_init_scale, nonlinearity=concat_elu,
                 pos_emb_init=0.01):  # TODO this number
        super().__init__()

        in_channels, height, width = img_shape

        self.pos_emb = Parameter(torch.Tensor(hidden_channels, height, width))
        torch.nn.init.normal_(self.pos_emb, mean=0., std=pos_emb_init)

        self.proj_in = WnConv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.blocks = ModuleList([
            ConvAttnBlock_Imagenet32(channels=hidden_channels, aux_channels=aux_channels, attn_heads=attn_heads,
                                     pdrop=pdrop)
            for _ in range(blocks)
        ])

        # additional nonlinearity added in compared to CIFAR
        self.nonlinearity = nonlinearity

        self.proj_out = WnConv2d(in_channels=hidden_channels * 2, out_channels=out_channels,
                                 kernel_size=3, padding=1, init_scale=output_init_scale)

    def forward(self, x, *, aux):
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x, aux=aux, pos_emb=self.pos_emb)
        x = self.nonlinearity(x)
        x = self.proj_out(x)
        return x


class MixLogisticConvAttnCoupling_Imagenet32(BaseFlow):
    def __init__(self, *, cf_shape, hidden_channels: int, aux_channels: int,
                 blocks: int, mix_components: int, attn_heads: int, pdrop: float,
                 force_float32_cond: bool):
        # force_float32_cond: if True, then the conditioning network is always run in float32, even if the data being
        # flowed through the model is in float64.
        super().__init__()
        in_channels, _, _ = cf_shape
        self.mix_components = mix_components
        self.num_trans_params = 2 + 3 * mix_components  # 2 for affine, 3*mix_components for logistic mixture
        self.cond = ConvAttnStack_Imagenet32(
            img_shape=cf_shape, hidden_channels=hidden_channels, out_channels=in_channels * self.num_trans_params,
            aux_channels=aux_channels, blocks=blocks, attn_heads=attn_heads, pdrop=pdrop, output_init_scale=0.1
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

        # add in a max with -7 compared to CIFAR10
        logscales = torch.clamp(logscales, min=-7)

        assert aff_logscale.shape == aff_translation.shape == cf.shape
        assert logits.shape == means.shape == logscales.shape == (bs, self.mix_components, *cf.shape[1:])
        # Make the flow
        return Compose([
            MixLogisticCDF(logits=logits, means=means, logscales=logscales, mix_dim=1),
            Inverse(Sigmoid()),
            ElementwiseAffine(logscales=aff_logscale, translations=aff_translation)
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
