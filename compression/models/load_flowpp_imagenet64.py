from collections import defaultdict, OrderedDict

import numpy as np
import torch
from torch.nn import Module, ModuleList

from compression.bitstream import CompressionModel
from compression.coupling import GatedConv
from compression.coupling import (
    Parallel, TupleFlip, Squeeze, StripeSplit, ChannelSplit, ImgLayerNorm
)
from compression.flows import BaseFlow, Compose, Inverse, ImgProc, Normalize, Sigmoid
from compression.nn import Conv2d
from compression.utils import sumflat, standard_normal_logp
from .flows_imagenet64 import MixLogisticConvAttnCoupling_Imagenet64


class Imagenet64Model(CompressionModel):
    def __init__(self, *, hdim=96, blocks=16, dequant_blocks=5, mix_components=4, attn_heads=4, pdrop=0.,
                 force_float32_cond):

        def coupling(cf_shape_, for_dequant=False, attn_version=True):
            return [
                Parallel([lambda: Normalize(cf_shape_)] * 2),
                MixLogisticConvAttnCoupling_Imagenet64(
                    cf_shape=cf_shape_,
                    hidden_channels=hdim,
                    aux_channels=32 if for_dequant else 0,
                    blocks=dequant_blocks if for_dequant else blocks,
                    mix_components=mix_components,
                    attn_heads=attn_heads,
                    pdrop=pdrop,
                    force_float32_cond=force_float32_cond,
                    attn_version=attn_version
                ),
                TupleFlip(),
            ]

        class Dequant(BaseFlow):
            def __init__(self):
                super().__init__()

                class DeepProcessor(Module):
                    def __init__(self):
                        super().__init__()
                        hdim2 = 32
                        self.conv = Conv2d(in_channels=6, out_channels=32, kernel_size=3, padding=1)
                        self.gatedconvs = ModuleList([])
                        self.norm1 = ModuleList([])
                        for _ in range(dequant_blocks):
                            self.gatedconvs.append(
                                GatedConv(in_channels=hdim2, aux_channels=0, gate_nin=False, pdrop=pdrop))
                            self.norm1.append(ImgLayerNorm(hdim2))

                    def forward(self, x):
                        processed_context = self.conv(x)
                        for i in range(len(self.gatedconvs)):
                            processed_context = self.gatedconvs[i](processed_context, aux=None)
                            processed_context = self.norm1[i](processed_context)
                        return processed_context

                self.context_proc = DeepProcessor()

                self.noise_flow = Compose([
                    # input: Gaussian noise
                    StripeSplit(),
                    *coupling((3, 64, 32), for_dequant=True, attn_version=False),
                    *coupling((3, 64, 32), for_dequant=True, attn_version=False),
                    *coupling((3, 64, 32), for_dequant=True, attn_version=False),
                    *coupling((3, 64, 32), for_dequant=True, attn_version=False),
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
                # input image 3, 64, 64
                ImgProc(),

                Squeeze(),
                # 12, 32, 32

                StripeSplit(),
                *coupling((12, 32, 16)),
                *coupling((12, 32, 16)),
                *coupling((12, 32, 16)),
                *coupling((12, 32, 16)),
                Inverse(StripeSplit()),

                # 12, 32, 32
                Squeeze(),
                # 48, 16, 16

                ChannelSplit(),
                *coupling((24, 16, 16)),
                *coupling((24, 16, 16)),
                Inverse(ChannelSplit()),

                StripeSplit(),
                *coupling((48, 16, 8)),
                *coupling((48, 16, 8)),
                Inverse(StripeSplit()),

                # 48, 16, 16
                Squeeze(),  # 192, 8, 8

                ChannelSplit(),
                *coupling((96, 8, 8)),
                *coupling((96, 8, 8)),
                Inverse(ChannelSplit()),

                StripeSplit(),
                *coupling((192, 8, 4)),
                *coupling((192, 8, 4)),
                Inverse(StripeSplit()),
            ]),
            dequant_flow=Dequant(),
            x_shape=(3, 64, 64),
            z_shape=(192, 8, 8)
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

    def set_debug_print(self, b):
        self.main_flow.dbgprint = b
        self.dequant_flow.noise_flow.dbgprint = b

    def load_from_tf(self, filename):
        tf_params = np.load(filename)
        torch_params = OrderedDict(sorted(list(self.named_parameters())))

        _unused_torch_names = set(torch_params.keys())
        _unused_tf_names = set(tf_params.keys())

        assert len(_unused_torch_names) == len(_unused_tf_names)
        from tqdm import tqdm
        bar = tqdm(list(range(len(_unused_torch_names))), desc='Loading parameters', leave=False)

        def load(torch_name, tf_name, transform, ema=True):
            if ema:
                tf_name += '/ExponentialMovingAverage'
            tensor = torch.from_numpy(tf_params[tf_name])
            if transform is not None:
                tensor = transform(tensor)
            torch_params[torch_name].data.copy_(tensor)
            _unused_torch_names.remove(torch_name)
            _unused_tf_names.remove(tf_name)
            bar.update()

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
            load(f'{torch_prefix}.bias', f'{tf_prefix}/b', None)
            load(f'{torch_prefix}.weight', f'{tf_prefix}/g', None)

        def load_conv_attn_block(torch_prefix, tf_prefix, aux, attn_version):
            if attn_version:
                load_gated_attn_block(torch_prefix + '.attn', tf_prefix + '/attn')
            if aux:
                load_dense(f'{torch_prefix}.conv.aux_proj.dense', f'{tf_prefix}/conv/a_proj')
            load_conv(f'{torch_prefix}.conv.conv', f'{tf_prefix}/conv/c1')
            load_dense(f'{torch_prefix}.conv.gate.nin.dense', f'{tf_prefix}/conv/c2')
            load_ln(f'{torch_prefix}.ln1.layernorm', f'{tf_prefix}/ln1')
            if attn_version:
                load_ln(f'{torch_prefix}.ln2.layernorm', f'{tf_prefix}/ln2')

        def load_gated_attn_block(torch_prefix, tf_prefix):
            load_dense(f'{torch_prefix}.proj_in.dense', f'{tf_prefix}/proj1')
            load_dense(f'{torch_prefix}.gate.nin.dense', f'{tf_prefix}/proj2')

        tf_counters = defaultdict(lambda: 0)

        def get_tf_counter(prefix):
            return prefix if (tf_counters[prefix] == 0) else f'{prefix}_{tf_counters[prefix]}'

        def load_coupling(prefix, i, blocks, aux, attn_version=True):
            tf_name = "MixLogisticAttnCoupling" if attn_version else "MixLogisticCoupling"
            load_norm(f'{prefix}.{i}.flows.0', f'{get_tf_counter("Norm")}/norm0')
            load_norm(f'{prefix}.{i}.flows.1', f'{get_tf_counter("Norm")}/norm1')
            tf_counters['Norm'] += 1

            load(f'{prefix}.{i + 1}.cond.pos_emb',
                 f'{get_tf_counter(tf_name)}/pos_emb', lambda t: t.permute(2, 0, 1))
            load_conv(f'{prefix}.{i + 1}.cond.proj_in', f'{get_tf_counter(tf_name)}/c1')
            load_conv(f'{prefix}.{i + 1}.cond.proj_out', f'{get_tf_counter(tf_name)}/c2')

            for block in range(blocks):
                load_conv_attn_block(f'{prefix}.{i + 1}.cond.blocks.{block}', f'{get_tf_counter(tf_name)}/block{block}',
                                     aux=aux, attn_version=attn_version)

            tf_counters[tf_name] += 1

        load_conv('dequant_flow.context_proc.conv', 'context_proc/proj')
        for i in range(5):
            load_gated_conv(f'dequant_flow.context_proc.gatedconvs.{i}', f'context_proc/c{i}')
            load_ln(f'dequant_flow.context_proc.norm1.{i}.layernorm', f'context_proc/dqln{i}')

        torch_dequant_inds = [1, 4, 7, 10]
        torch_dqn_pref = 'dequant_flow.noise_flow.flows'
        for torch_ind in torch_dequant_inds:
            load_coupling(torch_dqn_pref, torch_ind, 5, True, attn_version=False)

        # main flow
        torch_main_inds = [3, 6, 9, 12, 18, 21, 26, 29, 35, 38, 43, 46]
        torch_main_pref = 'main_flow.flows'
        for torch_ind in torch_main_inds:
            load_coupling(torch_main_pref, torch_ind, 16, False, attn_version=True)

        bar.close()
        assert len(_unused_tf_names) == len(_unused_torch_names) == 0
        return self


def load_imagenet64_model(filename, force_float32_cond, float32=False):
    model = Imagenet64Model(force_float32_cond=force_float32_cond).load_from_tf(filename).eval()
    if not float32:
        model = model.double()
    # freeze the model
    for p in model.parameters():
        p.requires_grad = False
    return model
