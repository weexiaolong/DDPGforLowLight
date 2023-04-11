from einops import rearrange
from typing import Optional, List
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GeLu(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))
        return x * cdf

class PreNorm(nn.Module):
    def __init__(self, dim, fn, layer=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.pos = None

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        if self.pos is not None:
            pos = rearrange(self.pos, 'b n (h d) -> b h n d', h=h)
            q += pos
            k += pos

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Attention_Dec(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., out=None):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.pos = None

        self.to_qv = nn.Linear(dim, dim * 2, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if out is  None else nn.Sequential(
            nn.Linear(dim, out),
            nn.Dropout(dropout)
        )

    def forward(self, x, tar, mask=None):
        b, n, _, h = *x.shape, self.heads
        qvk = self.to_qv(x).chunk(2, dim=-1)
        qvk += (self.to_k(tar),)
        q, v, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qvk)
        if self.pos is not None:
            pos = rearrange(self.pos, 'b n (h d) -> b h n d', h=h)
            q += pos
            k += pos

        dots = torch.einsum('bhid,bhjd->bhij', k, q) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GeLu(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, out_dim=None):

        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
        self.out = nn.Sequential(
            GeLu(),
            nn.Linear(dim, out_dim),
            nn.Dropout(dropout)
        ) if out_dim is not None else None

    def forward(self, x, mask=None, pos=None):

        for attn, ff in self.layers:
            if pos is not None:
                for m in attn.modules():
                    if isinstance(m, Attention):
                        setattr(m, 'pos', pos)
            x = attn(x, mask=mask)
            x = ff(x)
        if self.out is not None:
            x = self.out(x)
        return x


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class PositionEmbeddingSine(nn.Module):

    def __init__(self, num_pos_feats=256, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        not_mask = torch.ones_like(x[:, 0])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Actor(nn.Module):

    def __init__(self, number_f=24, depth=1, heads=8, dropout=0, patch_num=32):
        super(Actor, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 2, 1, bias=True)

        self.patch_num = patch_num
        self.transformer = Transformer(number_f, depth, heads, number_f, dropout)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

    def forward(self, x, img_in=True):
        img_in = x
        n, c, h, w = x.shape
        x_ = F.interpolate(x, (256, 256))

        x1 = self.e_conv1(x_)
        x1 = self.relu(x1)
        x1 = nn.AdaptiveAvgPool2d((32, 32))(x1)
        trans_inp = rearrange(x1, 'b c  h w -> b (  h w)  c ')
        x_out = self.transformer(trans_inp)
        x_r = rearrange(x_out, 'b (h w) c -> b  c  h w', h=self.patch_num)
        x_r = F.upsample_bilinear(x_r, (h, w)).tanh()
        x_r_resize = F.interpolate(x_r, x.shape[2:], mode='bilinear')

        return x_r_resize

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 16 * 16, 128)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = x.contiguous().view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))

        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv21 = nn.Conv2d(in_channels=24, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv1 = LeNet()
        self.conv2 = LeNet()

        self.fc1 = nn.Linear(128*2, 128)
        self.fc2 = nn.Linear(128, 1)

        self.train()

    def forward(self, x, a):
        x = self.conv1(x)

        a = F.relu(self.conv21(a))
        a = self.conv2(a)

        x = torch.cat((x, a), 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x