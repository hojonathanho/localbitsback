from collections import defaultdict, OrderedDict

import numpy as np
import torch
from torch.nn import Module, Parameter, ModuleList

from compression.bitstream import CompressionModel
from compression.coupling import (
    Parallel, TupleFlip, Squeeze, StripeSplit, ChannelSplit, ImgLayerNorm
)
from compression.flows import BaseFlow, Compose, Inverse, ImgProc, Normalize, Sigmoid
from compression.nn import WnConv2d
from compression.utils import sumflat, standard_normal_logp
from .flows_imagenet32 import MixLogisticConvAttnCoupling_Imagenet32, GatedConv_Imagenet32, GatedAttention_Imagenet32


class Imagenet32Model(CompressionModel):
    def __init__(self, *, hdim=128, blocks=20, dequant_blocks=8, mix_components=32, attn_heads=4, pdrop=0.,
                 force_float32_cond):

        def coupling(cf_shape_, for_dequant=False):
            return [
                Parallel([lambda: Normalize(cf_shape_)] * 2),
                MixLogisticConvAttnCoupling_Imagenet32(
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

                class DeepProcessor(Module):
                    def __init__(self):
                        super().__init__()

                        hdim2 = 32
                        height = width = 32
                        pos_emb_init = 0.01
                        self.pos_emb = Parameter(torch.Tensor(hdim2, height, width // 2))
                        torch.nn.init.normal_(self.pos_emb, mean=0., std=pos_emb_init)
                        self.conv = WnConv2d(in_channels=6, out_channels=32, kernel_size=3, padding=1)

                        self.gatedconvs = ModuleList([])
                        self.norm1 = ModuleList([])
                        self.gatedattns = ModuleList([])
                        self.norm2 = ModuleList([])
                        for _ in range(dequant_blocks):
                            self.gatedconvs.append(
                                GatedConv_Imagenet32(in_channels=hdim2, aux_channels=0, gate_nin=False, pdrop=pdrop))
                            self.norm1.append(ImgLayerNorm(hdim2))
                            self.gatedattns.append(
                                GatedAttention_Imagenet32(in_channels=hdim2, heads=attn_heads, pdrop=pdrop))
                            self.norm2.append(ImgLayerNorm(hdim2))

                    def forward(self, x):
                        processed_context = self.conv(x)
                        for i in range(len(self.gatedconvs)):
                            processed_context = self.gatedconvs[i](processed_context, aux=None)
                            processed_context = self.norm1[i](processed_context)
                            processed_context = self.gatedattns[i](processed_context, pos_emb=self.pos_emb)
                            processed_context = self.norm2[i](processed_context)
                        return processed_context

                self.context_proc = DeepProcessor()
                self.noise_flow = Compose([
                    # input: Gaussian noise
                    StripeSplit(),
                    *coupling((3, 32, 16), for_dequant=True),
                    *coupling((3, 32, 16), for_dequant=True),
                    *coupling((3, 32, 16), for_dequant=True),
                    *coupling((3, 32, 16), for_dequant=True),
                    Inverse(StripeSplit()),
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

                StripeSplit(),
                *coupling((3, 32, 16)),
                *coupling((3, 32, 16)),
                *coupling((3, 32, 16)),
                Inverse(StripeSplit()),

                Squeeze(),  # 12, 16, 16

                ChannelSplit(),
                *coupling((6, 16, 16)),
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
            load(f'{torch_prefix}.v', f'{tf_prefix}/V', lambda t: t.permute(1, 0))
            load(f'{torch_prefix}.g', f'{tf_prefix}/g', None)

        def load_conv(torch_prefix, tf_prefix):
            load(f'{torch_prefix}.b', f'{tf_prefix}/b', None)
            load(f'{torch_prefix}.v', f'{tf_prefix}/V', lambda t: t.permute(3, 2, 0, 1))
            load(f'{torch_prefix}.g', f'{tf_prefix}/g', None)

        def load_gated_conv(torch_prefix, tf_prefix):
            load_conv(f'{torch_prefix}.conv', f'{tf_prefix}/c1')
            load_conv(f'{torch_prefix}.gate.conv', f'{tf_prefix}/c2')

        def load_norm(torch_prefix, tf_prefix):
            load(f'{torch_prefix}.normalize.b', f'{tf_prefix}/b', lambda t: t.permute(2, 0, 1))
            load(f'{torch_prefix}.normalize.g', f'{tf_prefix}/g', lambda t: t.permute(2, 0, 1))

        def load_ln(torch_prefix, tf_prefix):
            load(f'{torch_prefix}.bias', f'{tf_prefix}/b', None)
            load(f'{torch_prefix}.weight', f'{tf_prefix}/g', None)

        def load_conv_attn_block(torch_prefix, tf_prefix, aux):
            load_gated_attn_block(torch_prefix + '.attn', tf_prefix + '/attn')
            if aux:
                load_dense(f'{torch_prefix}.conv.aux_proj.dense', f'{tf_prefix}/conv/a_proj')
            load_conv(f'{torch_prefix}.conv.conv', f'{tf_prefix}/conv/c1')
            load_dense(f'{torch_prefix}.conv.gate.nin.dense', f'{tf_prefix}/conv/c2')
            load_ln(f'{torch_prefix}.ln1.layernorm', f'{tf_prefix}/ln1')
            load_ln(f'{torch_prefix}.ln2.layernorm', f'{tf_prefix}/ln2')

        def load_gated_attn_block(torch_prefix, tf_prefix):
            load_dense(f'{torch_prefix}.proj_in.dense', f'{tf_prefix}/proj1')
            load_dense(f'{torch_prefix}.gate.nin.dense', f'{tf_prefix}/proj2')

        tf_counters = defaultdict(lambda: 0)

        def get_tf_counter(prefix):
            return prefix if (tf_counters[prefix] == 0) else f'{prefix}_{tf_counters[prefix]}'

        def load_coupling(prefix, i, blocks, aux):
            load_norm(f'{prefix}.{i}.flows.0', f'{get_tf_counter("Norm")}/norm0')
            load_norm(f'{prefix}.{i}.flows.1', f'{get_tf_counter("Norm")}/norm1')
            tf_counters['Norm'] += 1

            load(f'{prefix}.{i + 1}.cond.pos_emb',
                 f'{get_tf_counter("MixLogisticAttnCoupling")}/pos_emb', lambda t: t.permute(2, 0, 1))
            load_conv(f'{prefix}.{i + 1}.cond.proj_in',
                      f'{get_tf_counter("MixLogisticAttnCoupling")}/c1')
            load_conv(f'{prefix}.{i + 1}.cond.proj_out',
                      f'{get_tf_counter("MixLogisticAttnCoupling")}/c2')

            for block in range(blocks):
                load_conv_attn_block(f'{prefix}.{i + 1}.cond.blocks.{block}',
                                     f'{get_tf_counter("MixLogisticAttnCoupling")}/block{block}', aux=aux)
            tf_counters['MixLogisticAttnCoupling'] += 1

        # context proc
        load('dequant_flow.context_proc.pos_emb', 'context_proc/pos_emb_dq', lambda t: t.permute(2, 0, 1))
        load_conv('dequant_flow.context_proc.conv', 'context_proc/proj')
        for i in range(8):
            load_gated_conv(f'dequant_flow.context_proc.gatedconvs.{i}', f'context_proc/c{i}')
            load_ln(f'dequant_flow.context_proc.norm1.{i}.layernorm', f'context_proc/dqln{i}')
            load_gated_attn_block(f'dequant_flow.context_proc.gatedattns.{i}', f'context_proc/dqattn{i}')
            load_ln(f'dequant_flow.context_proc.norm2.{i}.layernorm', f'context_proc/ln{i}')

        torch_dequant_inds = [1, 4, 7, 10, 15, 18, 21, 24]
        torch_dqn_pref = 'dequant_flow.noise_flow.flows'
        for torch_ind in torch_dequant_inds:
            load_coupling(torch_dqn_pref, torch_ind, 8, True)

        # main flow
        torch_main_inds = [2, 5, 8, 11, 16, 19, 22, 28, 31, 34, 39, 42, 45]
        torch_main_pref = 'main_flow.flows'
        for torch_ind in torch_main_inds:
            load_coupling(torch_main_pref, torch_ind, 20, False)

        bar.close()

        assert len(_unused_tf_names) == len(_unused_torch_names) == 0
        return self


def load_imagenet32_model(filename, force_float32_cond, float32=False):
    model = Imagenet32Model(force_float32_cond=force_float32_cond).load_from_tf(filename).eval()
    if not float32:
        model = model.double()
    # freeze the model
    for p in model.parameters():
        p.requires_grad = False
    return model
