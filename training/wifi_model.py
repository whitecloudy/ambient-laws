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
    if x.ndim == 4 and scale.ndim == 3:
        return x * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)
    return x * (1 + scale) + shift


@torch.jit.script
def dit_gate_add(x: torch.Tensor, gate: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return x + gate.unsqueeze(2) * out


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
        return torch.stack([table.cos(), table.sin()], dim=1)
    
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
        x = torch.stack([x.cos(), x.sin()], dim=1)
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
        rt = cm.complex_mul(x, self.embedding.to(device=x.device, dtype=x.dtype))
        return rt

    def _build_embedding(self, max_len, hidden_dim):
        steps = torch.arange(max_len).unsqueeze(1)  # [P,1]
        dims = torch.arange(hidden_dim).unsqueeze(0)          # [1,E]
        table = steps * torch.exp(-math.log(max_len)
                                  * dims / hidden_dim)     # [P,E]
        return torch.stack([table.cos(), table.sin()], dim=0).unsqueeze(0)


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
        - Input:
          x, [2, B, N, H]
          c, [2, B, H] (or [2, B, N, H])
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=2)
        mod_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = dit_gate_add(x, gate_msa, self.attn(mod_x, mod_x, mod_x))
        x = dit_gate_add(x, gate_mlp, self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
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
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
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
        self.cond_dim = cond_dim
        self.p_embed = PositionEmbedding(
            sample_rate, input_dim, hidden_dim)
        self.t_embed = DiffusionEmbedding(
            max_step, embed_dim, hidden_dim)
        # self.t_embed = PositionalEmbedding(num_channels=embed_dim, endpoint=True)

        if self.cond_dim != 0:
            self.c_embed = MLPConditionEmbedding(cond_dim, hidden_dim)
        else:
            self.c_embed = None

        self.blocks = nn.ModuleList([
            DiA(self.hidden_dim, self.num_heads, self.dropout, self.mlp_ratio) for _ in range(num_block)
        ])
        self.final_layer = FinalLayer(self.hidden_dim, self.output_dim)

    def forward(self, x, noise_labels, class_labels=None, augment_labels=None, **kwargs):
        """
        Args:
            x: Input tensor of shape [2, Batch_size, Sequence_length, Feature_dimension]
               or [2, Batch_size, Antenna, Sequence_length, Feature_dimension],
               or torch.complex64 tensor.
            noise_labels: Noise labels tensor.
            class_labels: Class labels tensor.
        """
        is_complex_input = torch.is_complex(x)
        is_legacy = False

        if is_complex_input:
            x = torch.stack([x.real, x.imag], dim=1)
        elif x.ndim >= 3 and x.shape[-1] == 2:
            if x.shape[1] != 2:
                is_legacy = True
                perm = [0, x.ndim - 1] + list(range(1, x.ndim - 1))
                x = x.permute(*perm)
            elif (x.ndim == 4 and x.shape[-2] == self.input_dim) or (x.ndim == 5 and x.shape[1] * x.shape[3] == self.input_dim):
                is_legacy = True
                perm = [0, x.ndim - 1] + list(range(1, x.ndim - 1))
                x = x.permute(*perm)

        # Convert class_labels if provided
        c = class_labels
        if c is not None:
            if torch.is_complex(c):
                c = torch.stack([c.real, c.imag], dim=1)
            elif c.ndim == 2:
                c = torch.stack([c, torch.zeros_like(c)], dim=1)
            elif c.ndim >= 3 and c.shape[-1] == 2 and c.shape[1] != 2:
                perm = [0, c.ndim - 1] + list(range(1, c.ndim - 1))
                c = c.permute(*perm)
            c = c.to(x.dtype)

        # Cast inputs to x's dtype to support dynamic precision (float16, bfloat16, float32)
        noise_labels = noise_labels.to(x.dtype)
            
        antenna_reshape = False

        if x.ndim == 5:
            B, _, A, S, F = x.shape
            # [B, 2, A, S, F] -> [B, 2, S, A, F] -> [B, 2, S, A * F]
            x = x.permute(0, 1, 3, 2, 4).reshape(B, 2, S, A * F)
            antenna_reshape = True
            if noise_labels.shape == (B, 2, A, S, F):
                noise_labels = noise_labels.permute(0, 1, 3, 2, 4).reshape(B, 2, S, A * F)
            elif noise_labels.shape == (B, A, S, F):
                noise_labels = noise_labels.permute(0, 2, 1, 3).reshape(B, S, A * F)
        
        if self.dynamic_noise:
            if noise_labels.shape[0] == x.shape[0]:
                noise_labels = noise_labels.unsqueeze(1)
            while noise_labels.ndim < x.ndim:
                noise_labels = noise_labels.unsqueeze(-1)
                
            if noise_labels.shape != x.shape:
                noise_labels = noise_labels.expand_as(x)
        else:   # If not dynamic noise but when we receive dynamic noise label
            if noise_labels.ndim > 1:   # Expect only [Batch, ]
                noise_labels = torch.sqrt(torch.mean(torch.flatten(noise_labels, start_dim=1)**2, dim=1))

        t = noise_labels

        x = self.p_embed(x)
        t = self.t_embed(t)
        if self.c_embed is not None and c is not None:
            c = self.c_embed(c)
            c = c + t
        else:
            c = t
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)

        if antenna_reshape:
            # [B, 2, S, A * F] -> [B, 2, S, A, F] -> [B, 2, A, S, F]
            x = x.view(B, 2, S, A, F).permute(0, 1, 3, 2, 4)
        
        if is_complex_input:
            return torch.complex(x[:, 0], x[:, 1])
        if is_legacy:
            perm = [0] + list(range(2, x.ndim)) + [1]
            return x.permute(*perm)
        return x