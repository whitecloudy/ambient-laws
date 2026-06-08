import math
from math import sqrt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_utils import persistence

from .complex import complex_module as cm


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
    elif isinstance(module, cm.ComplexLinear):
        nn.init.xavier_uniform_(module.l_r.weight)
        nn.init.xavier_uniform_(module.l_i.weight)
        if module.l_r.bias is not None:
            nn.init.constant_(module.l_r.bias, 0)
            nn.init.constant_(module.l_i.bias, 0)


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
    
@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.view(-1)
        x = x.ger(freqs.to(x.dtype))
        x = torch.stack([x.cos(), x.sin()], dim=-1)
        return x



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
        # self.register_buffer('embedding', self._build_embedding(
        #     max_len, hidden_dim), persistent=False)
        embedding_tensor = self._build_embedding(max_len, hidden_dim)
        self.embedding = nn.Parameter(embedding_tensor, requires_grad=False)
        self.projection = cm.ComplexLinear(input_dim, hidden_dim)
        self.apply(init_weight_xavier)

    def forward(self, x): 
        x = self.projection(x)
        rt = cm.complex_mul(x, self.embedding.to(x.device).clone())
        return rt

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
        self.attn = cm.ComplexMultiHeadAttention(
            hidden_dim, hidden_dim, num_heads, dropout, bias=True, **block_kwargs)
        self.norm2 = cm.NaiveComplexLayerNorm(
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

    def forward(self, x, c):
        """
        Embedding diffusion step t with adaptive layer-norm.
        Embedding condition c with cross-attention.
        - Input:\\
          x, [B, N, H, 2], \\ 
          t, [B, H, 2], \\
          c, [B, N, H, 2], \\
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=1)
        mod_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + \
            gate_msa.unsqueeze(
                1) * self.attn(mod_x, mod_x, mod_x)
        x = x + \
            gate_mlp.unsqueeze(
                1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.linear = cm.ComplexLinear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 2*hidden_dim, bias=True)
        )
        self.apply(init_weight_zero)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class tfdiff_WiFi(nn.Module):
    def __init__(
        self,
        input_dim=90,
        hidden_dim=128,
        num_heads=8,
        sample_rate=512,
        max_step=10000,
        embed_dim=256,
        cond_dim=6,
        num_block=32,
        learn_tfdiff=False,
        dropout=0.0,
        mlp_ratio=4,
        label_dropout=0.0,
        dynamic_noise=False,
    ):
        super().__init__()
        self.learn_tfdiff = learn_tfdiff
        self.input_dim = input_dim
        self.output_dim = self.input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        self.dynamic_noise = dynamic_noise
        self.p_embed = PositionEmbedding(
            sample_rate, input_dim, hidden_dim)
        self.t_embed = DiffusionEmbedding(
            max_step, embed_dim, hidden_dim)
        # self.t_embed = PositionalEmbedding(num_channels=embed_dim, endpoint=True)

        self.c_embed = MLPConditionEmbedding(cond_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            DiA(self.hidden_dim, self.num_heads, self.dropout, self.mlp_ratio) for _ in range(num_block)
        ])
        self.final_layer = FinalLayer(self.hidden_dim, self.output_dim)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        """
        Args:
            x: Input tensor of shape [Batch_size, Sequence_length, Feature_dimension]
               or [Batch_size, Antenna, Sequence_length, Feature_dimension].
               (예: [B, sample_rate, input_dim] 또는 [B, A, sample_rate, F] 형태의 torch.complex64 텐서.
               4차원일 경우 Antenna 차원을 Feature_dimension으로 병합하여 [B, sample_rate, A * F]로 변환됨)
            noise_labels: Noise labels tensor.
            class_labels: Class labels tensor.
        """
        x = x
        
        antenna_reshape = False

        if x.ndim == 5:
            B, A, S, F, C = x.shape
            # [B, A, S, F] -> [B, S, A, F] -> [B, S, A * F]
            x = x.permute(0, 2, 1, 3, 4).reshape(B, S, A * F, C)
            antenna_reshape = True
            if noise_labels.shape == (B, A, S, F, C):
                noise_labels = noise_labels.permute(0, 2, 1, 3, 4).reshape(B, S, A * F, C)
        
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

        if c.ndim == 2:
            c = torch.stack([c, torch.zeros_like(c)], dim=-1)

        x = self.p_embed(x)
        t = self.t_embed(t)
        c = self.c_embed(c)
        c = c + t
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)

        if antenna_reshape:
            # [B, S, A * F] -> [B, S, A, F] -> [B, A, S, F]
            x = x.view(B, S, A, F, 2).permute(0, 2, 1, 3, 4)
        
        return x