import math
from math import sqrt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .complex import complex_module as cm
from .wifi_model import PositionalEmbedding


def init_weight_norm(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weight_zero(module):
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weight_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


@torch.jit.script
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_step, embed_dim=256, hidden_dim=256):
        super().__init__()
        # self.register_buffer('embedding', self._build_embedding(
        #     max_step, embed_dim), persistent=False)
        self.embedding = PositionalEmbedding(num_channels=embed_dim, endpoint=True)
        self.projection = nn.Sequential(
            cm.ComplexLinear(embed_dim, hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, hidden_dim, bias=True),
        )
        self.hidden_dim = hidden_dim
        self.apply(init_weight_norm)

    def forward(self, t):
        x = self.embedding(t)
        # if t.dtype in [torch.int32, torch.int64]:
        #     x = self.embedding[t]
        # else:
        #     x = self._lerp_embedding(t)
        return self.projection(x)

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_step, embed_dim):
        steps = torch.arange(max_step).unsqueeze(1)  # [T, 1]
        dims = torch.arange(embed_dim).unsqueeze(0)  # [1, E]
        table = steps * torch.exp(-math.log(max_step)
                                  * dims / embed_dim)  # [T, E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table

# TODO: Replace MLP with nn.Embedding
class MLPConditionEmbedding(nn.Module):
    def __init__(self, cond_dim, hidden_dim=256):
        super().__init__()
        self.projection = nn.Sequential(
            cm.ComplexLinear(cond_dim, hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, hidden_dim*4, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim*4, hidden_dim, bias=True),
        )
        self.apply(init_weight_norm)

    def forward(self, c):
        return self.projection(c)


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, input_dim, hidden_dim):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_len, hidden_dim), persistent=False)
        self.projection = cm.ComplexLinear(input_dim, hidden_dim)
        self.apply(init_weight_xavier)

    def forward(self, x):
        x = self.projection(x)
        return cm.complex_mul(x, self.embedding.to(x.device))

    def _build_embedding(self, max_len, hidden_dim):
        steps = torch.arange(max_len).unsqueeze(1)  # [P,1]
        dims = torch.arange(hidden_dim).unsqueeze(0)          # [1,E]
        table = steps * torch.exp(-math.log(max_len)
                                  * dims / hidden_dim)     # [P,E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table


class DiA(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.s_attn = cm.ComplexMultiHeadAttention(
            hidden_dim, hidden_dim, num_heads, dropout, bias=True, **block_kwargs)
        self.norm2 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.normc = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.x_attn = cm.ComplexMultiHeadAttention(
            hidden_dim, hidden_dim, num_heads, dropout, bias=True, *block_kwargs)
        self.norm3 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            cm.ComplexLinear(hidden_dim, mlp_hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(mlp_hidden_dim, hidden_dim, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 6*hidden_dim, bias=True)
        )
        self.apply(init_weight_xavier)
        self.adaLN_modulation.apply(init_weight_zero)

    def forward(self, x, t, c):
        """
        Embedding diffusion step t with adaptive layer-norm.
        Embedding condition c with cross-attention.
        - Input:\\
          x, [B, N, H, 2], \\ 
          t, [B, H, 2], \\
          c, [B, N, H, 2], \\
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            t).chunk(6, dim=1)
        mod_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + \
            gate_msa.unsqueeze(
                1) * self.s_attn(mod_x, mod_x, mod_x)
        x = x + self.x_attn(queries=self.normc(c),
                            keys=self.norm2(x), values=self.norm2(x))
        x = x + \
            gate_mlp.unsqueeze(
                1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 2*hidden_dim, bias=True)
        )
        self.linear = cm.ComplexLinear(hidden_dim, out_dim, bias=True)
        self.apply(init_weight_zero)

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = x + modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class SpatialDiffusion(nn.Module):
    """
    Process each sample of a sequence.
    Take CSI diffusion as an example.
    - Input:\\
      x, [B, S, A, 2], \\
      t, [B], \\
      c, [B, C, 2], \\
    - Output:
      n, [B, S*A, 2]
    """

    def __init__(self, num_spatial_block,
    input_dim,
    input_len,
    output_dim,
    hidden_dim,
    num_heads,
    max_step,
    embed_dim,
    cond_dim,
    dropout,
    mlp_ratio):
        super().__init__()
        self.num_block = num_spatial_block
        self.input_dim = input_dim
        self.input_len = input_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_step = max_step
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        self.p_embed = PositionEmbedding(
            self.input_len, self.input_dim, self.hidden_dim)
        self.t_embed = DiffusionEmbedding(
            self.max_step, self.embed_dim, self.hidden_dim)
        self.c_embed = MLPConditionEmbedding(self.cond_dim, self.hidden_dim)
        # A series of concatenated DiA blocks.
        self.blocks = nn.ModuleList([
            DiA(self.hidden_dim, self.num_heads, self.dropout, self.mlp_ratio) for _ in range(self.num_block)
        ])
        self.adaMLP = nn.Sequential(
            # Flatten [B, S, A, 2] to [B, S*A, 2]
            nn.Flatten(start_dim=1, end_dim=-2),
            cm.ComplexLinear(self.input_len*self.hidden_dim, self.output_dim),
            cm.ComplexSiLU(),
            cm.ComplexLinear(self.output_dim, self.output_dim),
        )
        self.adaMLP.apply(init_weight_xavier)

    def forward(self, x, t, c):
        x = self.p_embed(x)
        t = self.t_embed(t)
        c = self.c_embed(c)
        for block in self.blocks:
            x = block(x, t, c)
        x = self.adaMLP(x)
        return x



class tfdiff_mimo(nn.Module):
    """
    Signal Modulation and Augmentation via Generative Diffusion Model.
    Take CSI diffusion as an example.
    - Input:\\
      x, [B, N, S, A, 2], \\
      t, [B], \\
      c, [B, N, C, 2], \\
    - Output:
      n, [B, N, S, A, 2]
    """

    def __init__(self, 
        input_dim=[8, 26],
        hidden_dim=32,
        num_heads=4,
        sample_rate=1,
        max_step=10000,
        embed_dim=64,
        cond_dim=[8, 26],
        num_block=32,
        learn_tfdiff=False,
        dropout=0.1,
        mlp_ratio=4,
        label_dropout=0.0,
        dynamic_noise=False,):
        super().__init__()
        self.sample_rate = sample_rate
        self.extra_dim = [input_dim[0], input_dim[1]]
        self.cond_dim = [cond_dim[0], cond_dim[1]]
        self.dynamic_noise = dynamic_noise
        self.label_dropout = label_dropout

        # N parallel SpatialDiffusion blocks.
        self.spatial_block = SpatialDiffusion(
            num_spatial_block=num_block,
            input_dim=input_dim[1],  # 26 (Subcarrier) acts as feature dimension
            input_len=input_dim[0],  # 8 (Antenna) acts as sequence length (tokens)
            output_dim=input_dim[0]*input_dim[1],
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_step=max_step,
            embed_dim=embed_dim,
            cond_dim=cond_dim[1],    # 26 (Subcarrier) acts as feature dimension
            dropout=dropout,
            mlp_ratio=mlp_ratio
        )

    # N : number of samples in a frame. 
    # S : number of subcarriers.
    # A : number of antennas.
    # C : number of conditions.
    # x : [B, N, S, A, 2]
    # t : [B]
    # c : [B, N, C, 2]
    def forward(self, x, noise_labels, class_labels):
        x = x
        if self.dynamic_noise:
            while noise_labels.ndim < x.ndim:
                noise_labels = noise_labels.unsqueeze(-1)
                
            if noise_labels.shape != x.shape:
                noise_labels = noise_labels.expand_as(x)
            # dynamic_noise=True 일 때 noise_labels 최종 shape: [B, C, H, W]
        else:   #If not dynamic noise but when we receive dynamic noise label
            if noise_labels.ndim > 1:   # Expect only [Batch, ]
                noise_labels = torch.sqrt(torch.mean(torch.flatten(noise_labels, start_dim=1)**2, dim=1))
            # dynamic_noise=False 일 때 noise_labels 최종 shape: [B]

        t = noise_labels
        c = class_labels

        x_s = x.reshape([-1]+self.extra_dim+[2])  # [B*N, A, S, 2] (e.g. [B*N, 8, 26, 2])
        c_s = c.reshape([-1]+self.cond_dim+[2])  # [B*N, A, S, 2] (e.g. [B*N, 8, 26, 2])
        x_s = self.spatial_block(x_s, t.repeat_interleave(
            self.sample_rate), c_s)  # [B*N, A*S, 2]
        x = x_s.reshape([-1, self.sample_rate] +
                        self.extra_dim+[2])  # [B, N, A, S, 2] (e.g. [B, N, 8, 26, 2])
        return x
