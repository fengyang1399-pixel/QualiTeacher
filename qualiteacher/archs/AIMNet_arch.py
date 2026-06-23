"""
AIMNet architecture (from "Contrastive Semi-supervised Learning for Underwater Image Restoration via Reliable Bank", CVPR 2023).

This file is a "single-file arch.py style" consolidation of the official Semi-UIR repo:
- attention.py
- deform_conv.py
- model.py

Official repo: https://github.com/Huang-ShiRui/Semi-UIR

Colabator integration notes:
- Official AIMNet expects two inputs: (x, la) where la is a 3-channel illumination map.
- Official AIMNet returns two outputs: (result, res_grad).

We keep the official computation graph intact in `_forward_repo(x, la, score)`.
We add a Colabator-friendly `forward(...)` wrapper that:
- accepts aliases: input/x/img/lq
- accepts LA as `la=` or `depth=`
- accepts score= for quality-aware score injection
- tolerates extra kwargs (finetune/debug) used by Colabator
- returns ([result], [dummy_trans]) by default

Score injection:
- nn.Embedding(8, bot_ch) at bot level (128ch = n_feat * chan_factor^2)
- Injected twice: after Round 1 bot processing & after Round 2 bot processing
- Zero-initialized so pretrained weights are not disturbed at start
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

try:
    from basicsr.utils.registry import ARCH_REGISTRY
except Exception:
    ARCH_REGISTRY = None

# Use torchvision's (modulated) deformable conv instead of mmcv, which has no
# prebuilt wheels for recent PyTorch/CUDA (e.g. cu128 / Blackwell sm_120).
from torchvision.ops import deform_conv2d


# ===================== DCN_layer =====================

class DCN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, deformable_groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.conv_offset_mask = nn.Conv2d(
            in_channels * 2,
            deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size, stride=_pair(stride), padding=_pair(padding), bias=True)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.init_offset()
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input_feat, inter):
        feat = torch.cat([input_feat, inter], dim=1)
        out = self.conv_offset_mask(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(
            input_feat.contiguous(), offset, self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation, mask=mask)


# ===================== NonLocalSparseAttention =====================

def batched_index_select(values, indices):
    return values.gather(1, indices.unsqueeze(-1).expand(-1, -1, values.size(-1)))


class NonLocalSparseAttention(nn.Module):
    def __init__(self, n_hashes=4, channels=64, k_size=3, reduction=4, chunk_size=144, res_scale=1.0):
        super().__init__()
        self.chunk_size = chunk_size
        self.n_hashes = n_hashes
        self.reduction = reduction
        self.res_scale = res_scale
        self.conv_match = nn.Conv2d(channels, channels // reduction, k_size, padding=k_size // 2, bias=True)
        self.conv_assembly = nn.Conv2d(channels, channels, 1, padding=0, bias=True)

    def LSH(self, hash_buckets, x):
        N = x.shape[0]
        device = x.device
        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets // 2)
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(N, -1, -1, -1)
        rotated_vecs = torch.einsum("btf,bfhi->bhti", x, random_rotations)
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
        hash_codes = torch.argmax(rotated_vecs, dim=-1)
        offsets = torch.arange(self.n_hashes, device=device).reshape(1, -1, 1) * hash_buckets
        return (hash_codes + offsets).reshape(N, -1)

    def add_adjacent_buckets(self, x):
        x_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        x_fwd = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
        return torch.cat([x, x_back, x_fwd], dim=3)

    def forward(self, input):
        N, _, H, W = input.shape
        x_embed = self.conv_match(input).view(N, -1, H * W).permute(0, 2, 1)
        y_embed = self.conv_assembly(input).view(N, -1, H * W).permute(0, 2, 1)
        L, C = x_embed.shape[-2:]

        hash_buckets = min(L // self.chunk_size + (L // self.chunk_size) % 2, 128)
        hash_codes = self.LSH(hash_buckets, x_embed).detach()

        _, indices = hash_codes.sort(dim=-1)
        _, undo_sort = indices.sort(dim=-1)
        mod_indices = indices % L
        x_sorted = batched_index_select(x_embed, mod_indices)
        y_sorted = batched_index_select(y_embed, mod_indices)

        pad = self.chunk_size - L % self.chunk_size if L % self.chunk_size != 0 else 0
        x_att = x_sorted.reshape(N, self.n_hashes, -1, C)
        y_att = y_sorted.reshape(N, self.n_hashes, -1, C * self.reduction)
        if pad:
            x_att = torch.cat([x_att, x_att[:, :, -pad:, :].clone()], dim=2)
            y_att = torch.cat([y_att, y_att[:, :, -pad:, :].clone()], dim=2)

        x_att = x_att.reshape(N, self.n_hashes, -1, self.chunk_size, C)
        y_att = y_att.reshape(N, self.n_hashes, -1, self.chunk_size, C * self.reduction)

        x_match = F.normalize(x_att, p=2, dim=-1, eps=5e-5)
        x_match = self.add_adjacent_buckets(x_match)
        y_att = self.add_adjacent_buckets(y_att)

        raw_score = torch.einsum("bhkie,bhkje->bhkij", x_att, x_match)
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score)
        bucket_score = bucket_score.reshape(N, self.n_hashes, -1)

        ret = torch.einsum("bukij,bukje->bukie", score, y_att)
        ret = ret.reshape(N, self.n_hashes, -1, C * self.reduction)

        if pad:
            ret = ret[:, :, :-pad, :].clone()
            bucket_score = bucket_score[:, :, :-pad].clone()

        ret = batched_index_select(ret.reshape(N, -1, C * self.reduction), undo_sort)
        bucket_score = bucket_score.reshape(N, -1).gather(1, undo_sort)

        ret = ret.reshape(N, self.n_hashes, L, C * self.reduction)
        bucket_score = bucket_score.reshape(N, self.n_hashes, L, 1)
        probs = F.softmax(bucket_score, dim=1)
        ret = (ret * probs).sum(dim=1)

        return ret.permute(0, 2, 1).view(N, -1, H, W).contiguous() * self.res_scale + input


# ===================== Building blocks =====================

class SFT_layer(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv_gamma = nn.Sequential(nn.Conv2d(ch_in, ch_out, 1, bias=False), nn.LeakyReLU(0.1, True), nn.Conv2d(ch_out, ch_out, 1, bias=False))
        self.conv_beta  = nn.Sequential(nn.Conv2d(ch_in, ch_out, 1, bias=False), nn.LeakyReLU(0.1, True), nn.Conv2d(ch_out, ch_out, 1, bias=False))

    def forward(self, x, inter):
        return x * self.conv_gamma(inter) + self.conv_beta(inter)


class IGM(nn.Module):
    def __init__(self, ch_in, ch_out, ks):
        super().__init__()
        self.dcn = DCN_layer(ch_in, ch_out, ks, padding=(ks - 1) // 2, bias=False)
        self.sft = SFT_layer(ch_in, ch_out)

    def forward(self, x, inter):
        return x + self.dcn(x, inter) + self.sft(x, inter)


class GetGradientNopadding(nn.Module):
    def __init__(self):
        super().__init__()
        kv = torch.FloatTensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]]).unsqueeze(0).unsqueeze(0)
        kh = torch.FloatTensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(kh, requires_grad=False)
        self.weight_v = nn.Parameter(kv, requires_grad=False)

    def forward(self, inp):
        x_list = []
        for i in range(inp.shape[1]):
            xi = inp[:, i:i+1]
            xv = F.conv2d(xi, self.weight_v, padding=1)
            xh = F.conv2d(xi, self.weight_h, padding=1)
            x_list.append(torch.sqrt(xv ** 2 + xh ** 2 + 1e-6))
        return torch.cat(x_list, dim=1)


class Down(nn.Module):
    def __init__(self, ch, cf, bias=False):
        super().__init__()
        self.bot = nn.Sequential(nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
                                 nn.Conv2d(ch, int(ch * cf), 1, bias=bias))
    def forward(self, x): return self.bot(x)


class DownSample(nn.Module):
    def __init__(self, ch, sf, cf=2, ks=3):
        super().__init__()
        layers = []
        for _ in range(int(np.log2(sf))):
            layers.append(Down(ch, cf)); ch = int(ch * cf)
        self.body = nn.Sequential(*layers)
    def forward(self, x): return self.body(x)


class Up(nn.Module):
    def __init__(self, ch, cf, bias=False):
        super().__init__()
        self.bot = nn.Sequential(nn.Conv2d(ch, int(ch // cf), 1, bias=bias),
                                 nn.Upsample(scale_factor=2, mode="bilinear", align_corners=bias))
    def forward(self, x): return self.bot(x)


class UpSample(nn.Module):
    def __init__(self, ch, sf, cf=2, ks=3):
        super().__init__()
        layers = []
        for _ in range(int(np.log2(sf))):
            layers.append(Up(ch, cf)); ch = int(ch // cf)
        self.body = nn.Sequential(*layers)
    def forward(self, x): return self.body(x)


class ContextBlock(nn.Module):
    def __init__(self, n, act, bias=True):
        super().__init__()
        self.conv_mask = nn.Conv2d(n, 1, 1, bias=bias)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(nn.Conv2d(n, n, 1, bias=bias), act, nn.Conv2d(n, n, 1, bias=bias))

    def forward(self, x):
        B, C, H, W = x.size()
        ctx = self.softmax(self.conv_mask(x).view(B, 1, H * W)).unsqueeze(3)
        ctx = torch.matmul(x.view(B, C, H * W).unsqueeze(1), ctx).view(B, C, 1, 1)
        return x + self.channel_add_conv(ctx)


class RCB(nn.Module):
    def __init__(self, n, act, bias=True):
        super().__init__()
        self.act = act
        self.body = nn.Sequential(nn.Conv2d(n, n, 3, 1, 1, bias=bias), act, nn.Conv2d(n, n, 3, 1, 1, bias=bias))
        self.gcnet = ContextBlock(n, act, bias)

    def forward(self, x):
        return x + self.act(self.gcnet(self.body(x)))


class AFF(nn.Module):
    def __init__(self, ch, act, r=4):
        super().__init__()
        ic = ch // r
        self.local_att = nn.Sequential(nn.Conv2d(ch, ic, 1), act, nn.Conv2d(ic, ch, 1))
        self.global_att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, ic, 1), act, nn.Conv2d(ic, ch, 1))

    def forward(self, x, res):
        w = torch.sigmoid(self.local_att(x + res) + self.global_att(x + res))
        return 2 * x * w + 2 * res * (1 - w)


class AtrousBlock(nn.Module):
    def __init__(self, mc, ks, s, act, atrous=(1, 2, 3, 4)):
        super().__init__()
        self.atrous_layers = nn.Sequential(
            nn.Conv2d(mc, mc // 2, ks, s, dilation=atrous[0], padding=atrous[0]),
            nn.Conv2d(mc, mc // 2, ks, s, dilation=atrous[1], padding=atrous[1]),
            nn.Conv2d(mc, mc // 2, ks, s, dilation=atrous[2], padding=atrous[2]),
            nn.Conv2d(mc, mc // 2, ks, s, dilation=atrous[3], padding=atrous[3]))
        self.conv = nn.Conv2d(mc * 2, mc, 1)
        self.act = act
        self.att = AFF(mc, act)

    def forward(self, data):
        xs = [self.act(self.atrous_layers[i](data)) for i in range(4)]
        return self.att(data, self.act(self.conv(torch.cat(xs, 1))))


# ===================== AIMnet =====================

@(ARCH_REGISTRY.register() if ARCH_REGISTRY is not None else (lambda cls: cls))
class AIMnet(nn.Module):
    """
    AIM-Net (Semi-UIR, CVPR 2023) + Score Injection.

    Channel structure: top=32, mid=64, bot=128.
    Score injection: bot level (128ch), two rounds, zero-initialized.
    LA: pre-computed, passed in via la= or depth=.
    """

    def __init__(self, n_feat=32, height=256, width=256, n_RCB=2, chan_factor=2, bias=True):
        super().__init__()
        self.n_feat, self.height, self.width = n_feat, height, width
        act = nn.LeakyReLU(0.1, True)
        self.act = act
        atrous = [1, 2, 3, 4]

        self.dau_top = nn.Sequential(*[RCB(int(n_feat * chan_factor**0), act, bias) for _ in range(n_RCB)])
        self.dau_mid = nn.Sequential(*[RCB(int(n_feat * chan_factor**1), act, bias) for _ in range(n_RCB)])
        self.dau_bot = nn.Sequential(*[RCB(int(n_feat * chan_factor**2), act, bias) for _ in range(n_RCB)])

        self.atb_top = AtrousBlock(int(n_feat * chan_factor**0), 3, 1, act, atrous)
        self.atb_mid = AtrousBlock(int(n_feat * chan_factor**1), 3, 1, act, atrous)
        self.atb_bot = AtrousBlock(int(n_feat * chan_factor**2), 3, 1, act, atrous)

        self.nl_top = NonLocalSparseAttention(channels=int(n_feat * chan_factor**0))
        self.nl_mid = NonLocalSparseAttention(channels=int(n_feat * chan_factor**1))
        self.nl_bot = NonLocalSparseAttention(channels=int(n_feat * chan_factor**2))

        self.down2 = DownSample(int(chan_factor**0 * n_feat), 2, chan_factor)
        self.down4 = nn.Sequential(DownSample(int(chan_factor**0 * n_feat), 2, chan_factor),
                                   DownSample(int(chan_factor**1 * n_feat), 2, chan_factor))

        self.up21_1 = UpSample(int(chan_factor**1 * n_feat), 2, chan_factor)
        self.up21_2 = UpSample(int(chan_factor**1 * n_feat), 2, chan_factor)
        self.up32_1 = UpSample(int(chan_factor**2 * n_feat), 2, chan_factor)
        self.up32_2 = UpSample(int(chan_factor**2 * n_feat), 2, chan_factor)

        self.conv_in  = nn.Conv2d(3, n_feat, 3, padding=1, bias=bias)
        self.conv_mid = nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=bias)
        self.conv_out = nn.Conv2d(n_feat, 3, 3, padding=1, bias=bias)
        self.grad_out = nn.Conv2d(n_feat, 3, 1, padding=0, bias=bias)

        self.aff_top   = AFF(int(n_feat * chan_factor**0), act)
        self.aff_mid   = AFF(int(n_feat * chan_factor**1), act)
        self.aff_final = AFF(n_feat, act)
        self.igb_layer = IGM(n_feat, n_feat, 3)

        self.get_gradient = GetGradientNopadding()
        self.b_concat_1 = nn.Conv2d(2 * n_feat, n_feat, 3, padding=1, bias=bias)
        self.b_block_1  = RCB(2 * n_feat, act, bias)
        self.b_concat_2 = nn.Conv2d(2 * n_feat, n_feat, 3, padding=1, bias=bias)
        self.b_block_2  = RCB(2 * n_feat, act, bias)
        self.b_fea_conv = nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=bias)  # in official ckpt

        # --- Score injection: bot=128ch ---
        bot_ch = int(n_feat * chan_factor ** 2)  # 128
        self.score_embedding = nn.Embedding(8, bot_ch)
        nn.init.zeros_(self.score_embedding.weight)

    # ---------- core forward ----------
    def _forward_repo(self, x, la, score=None):
        x_top = self.conv_in(x.clone())
        x_top_la = self.conv_in(la)
        x_grad = self.get_gradient(x)
        x_mid = self.down2(x_top)
        x_bot = self.down4(x_top)

        # Round 1
        x_top1 = self.dau_top(self.igb_layer(self.atb_top(x_top), x_top_la))
        x_mid1 = self.dau_mid(self.atb_mid(x_mid))
        x_bot1 = self.dau_bot(self.atb_bot(x_bot))
        if score is not None:
            score_emb = self.score_embedding(score.view(-1)).unsqueeze(-1).unsqueeze(-1)
            x_bot1 = x_bot1 + score_emb
        x_mid1 = self.aff_mid(x_mid1, self.up32_1(x_bot1))
        x_top1 = self.aff_top(x_top1, self.up21_1(x_mid1))

        # Round 2
        x_top2 = self.dau_top(self.igb_layer(self.atb_top(x_top1), x_top_la))
        x_mid2 = self.dau_mid(self.nl_mid(x_mid1))
        x_bot2 = self.dau_bot(self.nl_bot(x_bot1))
        if score is not None:
            x_bot2 = x_bot2 + score_emb
        x_mid2 = self.aff_mid(x_mid2, self.up32_2(x_bot2))
        x_top2 = self.aff_top(x_top2, self.up21_2(x_mid2))

        mid_out = self.conv_mid(x_top2) + x_top

        x_b = self.conv_in(x_grad)
        x_cat_1 = self.b_concat_1(self.b_block_1(torch.cat([x_b, x_top1], 1)))
        x_cat_2 = self.b_concat_2(self.b_block_2(torch.cat([x_cat_1, x_top2], 1)))
        grad_out = x_cat_2 + x_b
        res_grad = self.grad_out(grad_out)
        result = self.conv_out(self.aff_final(mid_out, grad_out))
        return result, res_grad

    # ---------- Colabator wrapper ----------
    def forward(self, input=None, x=None, img=None, lq=None,
                depth=None, la=None, score=None,
                finetune=False, debug=False, **kwargs):
        if input is None:
            input = img if img is not None else (lq if lq is not None else x)
        if input is None:
            raise ValueError("AIMnet.forward: no input provided.")
        if la is None:
            la = depth
        if la is None:
            raise ValueError("AIMnet requires LA. Pass la= or depth=.")

        result, res_grad = self._forward_repo(input, la, score=score)
        dummy_trans = torch.ones_like(result[:, :1, :, :])
        if debug:
            return [result], [dummy_trans], [], []
        return [result], [dummy_trans]