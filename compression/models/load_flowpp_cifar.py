from collections import defaultdict, OrderedDict

import numpy as np
import torch

from compression.bitstream import CompressionModel
from compression.coupling import (
    Parallel, MixLogisticConvAttnCoupling, TupleFlip, Squeeze, StripeSplit, ChannelSplit, GatedConv
)
from compression.flows import BaseFlow, Compose, Inverse, ImgProc, Normalize, Sigmoid, Pointwise
from compression.nn import Conv2d
from compression.utils import sumflat, standard_normal_logp


class CifarModel(CompressionModel):
    def __init__(self, *, hdim=96, blocks=10, dequant_blocks=2, mix_components=32, attn_heads=4, pdrop=0.2,
                 force_float32_cond):

        def coupling(cf_shape_, for_dequant=False):
            return [
                Parallel([lambda: Normalize(cf_shape_)] * 2),
                Parallel([lambda: Pointwise(channels=cf_shape_[0])] * 2),
                MixLogisticConvAttnCoupling(
                    cf_shape=cf_shape_,
                    hidden_channels=hdim,
                    aux_channels=32 if for_dequant else 0,
                    blocks=dequant_blocks if for_dequant else blocks,
                    mix_components=mix_components,
                    attn_heads=attn_heads,
                    pdrop=pdrop,
                    force_float32_cond=force_float32_cond
                ),
                TupleFlip(),
            ]

        class Dequant(BaseFlow):
            def __init__(self):
                super().__init__()
                self.context_proc = torch.nn.Sequential(
                    Conv2d(in_channels=6, out_channels=32, kernel_size=3, padding=1),
                    GatedConv(in_channels=32, aux_channels=0, pdrop=pdrop),
                    GatedConv(in_channels=32, aux_channels=0, pdrop=pdrop),
                    GatedConv(in_channels=32, aux_channels=0, pdrop=pdrop),
                )
                self.noise_flow = Compose([
                    # input: Gaussian noise
                    StripeSplit(),
                    *coupling((3, 32, 16), for_dequant=True),
                    *coupling((3, 32, 16), for_dequant=True),
                    *coupling((3, 32, 16), for_dequant=True),
                    *coupling((3, 32, 16), for_dequant=True),
                    Inverse(StripeSplit()),
                    Sigmoid(),
                ])
                self.aux_split = StripeSplit()

            def _process_context(self, aux):
                a = aux / 256.0 - 0.5
                a = torch.cat(self.aux_split(a, inverse=False, aux=None)[0], dim=1)
                return self.context_proc(a)

            def forward(self, eps, *, aux, inverse: bool):
                # base distribution noise -> dequantization noise
                return self.noise_flow(eps, aux=self._process_context(aux), inverse=inverse)

            def code(self, input_sym, *, aux, inverse: bool, stream):
                return self.noise_flow.code(input_sym, aux=self._process_context(aux), inverse=inverse, stream=stream)

        super().__init__(
            main_flow=Compose([
                # input image 3, 32, 32
                ImgProc(),

                StripeSplit(),
                *coupling((3, 32, 16)),
                *coupling((3, 32, 16)),
                *coupling((3, 32, 16)),
                *coupling((3, 32, 16)),
                Inverse(StripeSplit()),

                Squeeze(),  # 12, 16, 16

                ChannelSplit(),
                *coupling((6, 16, 16)),
                *coupling((6, 16, 16)),
                Inverse(ChannelSplit()),

                StripeSplit(),
                *coupling((12, 16, 8)),
                *coupling((12, 16, 8)),
                *coupling((12, 16, 8)),
                Inverse(StripeSplit()),
            ]),
            dequant_flow=Dequant(),
            x_shape=(3, 32, 32),
            z_shape=(12, 16, 16)
        )

    def calc_dequant_noise(self, x):
        eps = torch.randn_like(x)
        u, dequant_logd = self.dequant_flow(eps=eps, aux=x, inverse=False)
        assert u.shape == x.shape and dequant_logd.shape == (x.shape[0],)
        return u, dequant_logd - sumflat(standard_normal_logp(eps))

    def forward(self, x, *, u=None, dequant_logd=None):
        assert (u is None) == (dequant_logd is None)
        if u is None:
            u, dequant_logd = self.calc_dequant_noise(x)
        assert u.shape == x.shape and dequant_logd.shape == (x.shape[0],)
        assert (u >= 0).all() and (u <= 1).all()

        z, main_logd = self.main_flow(x + u, aux=None, inverse=False)
        z_logp = sumflat(standard_normal_logp(z))
        total_logd = dequant_logd + main_logd + z_logp
        assert z.shape[0] == x.shape[0] and z.numel() == x.numel()
        assert main_logd.shape == dequant_logd.shape == total_logd.shape == z_logp.shape == (x.shape[0],)
        return {
            'u': u,
            'z': z,
            'total_logd': total_logd,
            'dequant_logd': dequant_logd,
        }

    def load_from_tf(self, filename):
        tf_params = np.load(filename)
        torch_params = OrderedDict(sorted(list(self.named_parameters())))

        _unused_torch_names = set(torch_params.keys())
        _unused_tf_names = set(tf_params.keys())
        assert len(_unused_torch_names) == len(_unused_tf_names)
        from tqdm import tqdm
        bar = tqdm(list(range(len(_unused_torch_names))), desc='Loading parameters', leave=False)

        def load(torch_name, tf_name, transform):
            tensor = torch.from_numpy(tf_params[tf_name])
            if transform is not None:
                tensor = transform(tensor)
            torch_params[torch_name].data.copy_(tensor)
            _unused_torch_names.remove(torch_name)
            _unused_tf_names.remove(tf_name)
            bar.update()
            # print(torch_name, '<--', tf_name)

        def load_dense(torch_prefix, tf_prefix):
            load(f'{torch_prefix}.b', f'{tf_prefix}/b', None)
            load(f'{torch_prefix}.w', f'{tf_prefix}/W', lambda t: t.permute(1, 0))

        def load_conv(torch_prefix, tf_prefix):
            load(f'{torch_prefix}.b', f'{tf_prefix}/b', None)
            load(f'{torch_prefix}.w', f'{tf_prefix}/W', lambda t: t.permute(3, 2, 0, 1))

        def load_gated_conv(torch_prefix, tf_prefix):
            load_conv(f'{torch_prefix}.conv', f'{tf_prefix}/c1')
            load_conv(f'{torch_prefix}.gate.conv', f'{tf_prefix}/c2')

        def load_norm(torch_prefix, tf_prefix):
            load(f'{torch_prefix}.normalize.b', f'{tf_prefix}/b', lambda t: t.permute(2, 0, 1))
            load(f'{torch_prefix}.normalize.g', f'{tf_prefix}/g', lambda t: t.permute(2, 0, 1))

        def load_ln(torch_prefix, tf_prefix):
            load(f'{torch_prefix}.bias', f'{tf_prefix}/b', lambda t: t[0, 0, 0, :])
            load(f'{torch_prefix}.weight', f'{tf_prefix}/g', lambda t: t[0, 0, 0, :])

        def load_conv_attn_block(torch_prefix, tf_prefix, aux):
            load_dense(f'{torch_prefix}.attn.proj_in.dense', f'{tf_prefix}/attn/proj1')
            load_dense(f'{torch_prefix}.attn.gate.nin.dense', f'{tf_prefix}/attn/proj2')
            if aux:
                load_dense(f'{torch_prefix}.conv.aux_proj.dense', f'{tf_prefix}/conv/a_proj')
            load_conv(f'{torch_prefix}.conv.conv', f'{tf_prefix}/conv/c1')
            load_dense(f'{torch_prefix}.conv.gate.nin.dense', f'{tf_prefix}/conv/c2')
            load_ln(f'{torch_prefix}.ln1.layernorm', f'{tf_prefix}/ln1')
            load_ln(f'{torch_prefix}.ln2.layernorm', f'{tf_prefix}/ln2')

        tf_counters = defaultdict(lambda: 0)

        def get_tf_counter(prefix):
            return prefix if (tf_counters[prefix] == 0) else f'{prefix}_{tf_counters[prefix]}'

        def load_coupling(prefix, i, blocks, aux):
            load_norm(f'{prefix}.{i}.flows.0', f'{get_tf_counter("Norm")}/norm0')
            load_norm(f'{prefix}.{i}.flows.1', f'{get_tf_counter("Norm")}/norm1')
            tf_counters['Norm'] += 1

            load(f'{prefix}.{i + 1}.flows.0.w', f'{get_tf_counter("Pointwise")}/W0', lambda t: t.permute(1, 0))
            load(f'{prefix}.{i + 1}.flows.1.w', f'{get_tf_counter("Pointwise")}/W1', lambda t: t.permute(1, 0))
            tf_counters['Pointwise'] += 1

            load(f'{prefix}.{i + 2}.cond.pos_emb',
                 f'{get_tf_counter("MixLogisticAttnCoupling")}/pos_emb', lambda t: t.permute(2, 0, 1))
            load_conv(f'{prefix}.{i + 2}.cond.proj_in',
                      f'{get_tf_counter("MixLogisticAttnCoupling")}/proj_in')
            load_conv(f'{prefix}.{i + 2}.cond.proj_out',
                      f'{get_tf_counter("MixLogisticAttnCoupling")}/proj_out')

            for block in range(blocks):
                load_conv_attn_block(f'{prefix}.{i + 2}.cond.blocks.{block}',
                                     f'{get_tf_counter("MixLogisticAttnCoupling")}/block{block}', aux=aux)
            tf_counters['MixLogisticAttnCoupling'] += 1

        # context proc
        load_conv('dequant_flow.context_proc.0', 'context_proc/proj')
        for i in [1, 2, 3]:
            load_gated_conv(f'dequant_flow.context_proc.{i}', f'context_proc/c{i - 1}')
        # dequant flow
        for i in range(1, 15 + 1, 4):
            load_coupling('dequant_flow.noise_flow.flows', i, blocks=2, aux=True)
        # main flow
        for i in (list(range(2, 16 + 1, 4)) + list(range(21, 27 + 1, 4)) + list(range(31, 41 + 1, 4))):
            load_coupling('main_flow.flows', i, blocks=10, aux=False)

        bar.close()
        assert len(_unused_tf_names) == len(_unused_torch_names) == 0
        return self


def load_cifar_model(filename, force_float32_cond, float32=False):
    model = CifarModel(force_float32_cond=force_float32_cond).load_from_tf(filename).eval()
    if not float32:
        model = model.double()
    # freeze the model
    for p in model.parameters():
        p.requires_grad = False
    return model
