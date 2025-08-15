from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Config (JSON-parity)
# --------------------------------------------------------------------------------------

@dataclass
class RopeScalingConfig:
    """Configuration for RoPE scaling methods like YaRN."""
    factor: float = 32.0
    beta_slow: float = 1.0
    beta_fast: float = 32.0

@dataclass
class ModelConfig:
    """Defines the complete architecture of the GPT-OSS model."""
    # Core dimensions
    vocab_size: int = 201088
    hidden_size: int = 2880
    num_hidden_layers: int = 36
    intermediate_size: int = 2880  # FFN size inside each expert
    enable_sink_token: bool = False

    # Attention
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    attention_bias: bool = True
    attention_dropout: float = 0.0

    # Context & Position
    max_position_embeddings: int = 131_072
    sliding_window: int = 128
    layer_types: Optional[List[Literal["sliding_attention", "full_attention"]]] = None

    # MoE
    num_local_experts: int = 128
    experts_per_token: int = 4
    router_aux_loss_coef: float = 0.9

    # RoPE
    rope_theta: float = 150_000.0
    rope_scaling: RopeScalingConfig = field(default_factory=RopeScalingConfig)

    # Other
    hidden_act: str = "silu"
    swiglu_limit: float = 7.0
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    dropout: float = 0.0
    eos_token_id: Optional[int] = None

    def __post_init__(self):
        """Post-initialization checks and setup."""
        if self.layer_types is None:
            # Default to alternating SWA and Full Attention
            self.layer_types = [
                "sliding_attention" if i % 2 == 0 else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        assert len(self.layer_types) == self.num_hidden_layers
        assert self.num_attention_heads % self.num_key_value_heads == 0

    @property
    def group_size(self) -> int:
        """Number of query heads per key/value head in GQA."""
        return self.num_attention_heads // self.num_key_value_heads


# --------------------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


def swiglu(x: torch.Tensor, limit: Optional[float] = None) -> torch.Tensor:
    # x: (..., 2*F)
    up, gate = x.chunk(2, dim=-1)
    if limit is not None:
        up = up.clamp(-limit, limit)
        gate = gate.clamp(-limit, limit)
    return F.silu(gate) * up


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float, rope_scaling: RopeScalingConfig, device=None, dtype=None):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        inv_freq = 1.0 / (
            (rope_theta) ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq_base", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _update_cache(self, seqlen: int, device, dtype):
        if seqlen <= self._seq_len_cached and self.cos_cached.device == device and self.cos_cached.dtype == dtype:
            return
        # Simple YaRN-style scaling: stretch positions by factor
        # (This is a pragmatic approximation; exact HF equations can be dropped in here.)
        factor = float(self.rope_scaling.factor)
        pos = torch.arange(seqlen, device=device, dtype=torch.float32)
        # Blend two bands using beta_slow/beta_fast (lightweight proxy)
        beta_slow = float(self.rope_scaling.beta_slow)
        beta_fast = float(self.rope_scaling.beta_fast)
        # Effective inv_freq as convex mix around the band index
        inv_freq = self.inv_freq_base
        inv_freq_slow = inv_freq / max(1.0, beta_slow)
        inv_freq_fast = inv_freq / max(1.0, beta_fast)
        inv_freq_eff = 0.5 * (inv_freq_slow + inv_freq_fast)
        # Scale positions to expand context by `factor`
        pos_scaled = pos / factor
        freqs = torch.einsum("i,j->ij", pos_scaled, inv_freq_eff)
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        cos = torch.stack([cos, cos], dim=-1).reshape(seqlen, -1)
        sin = torch.stack([sin, sin], dim=-1).reshape(seqlen, -1)
        self.cos_cached = cos
        self.sin_cached = sin
        self._seq_len_cached = seqlen

    def apply_rotary(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H_head) where H_head == head_dim
        B, T, Dh = x.shape
        device, dtype = x.device, x.dtype
        self._update_cache(seqlen=int(positions.max().item()) + 1, device=device, dtype=dtype)
        cos = self.cos_cached[positions]  # (B, T, H_head)
        sin = self.sin_cached[positions]
        x1, x2 = x[..., ::2], x[..., 1::2]
        xr = torch.stack([x1 * cos[..., ::2] - x2 * sin[..., ::2],
                          x1 * sin[..., ::2] + x2 * cos[..., ::2]], dim=-1)
        return xr.flatten(-2)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        H = config.hidden_size
        self.n_head = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_embd = H
        self.q_proj = nn.Linear(H, self.n_head * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(H, self.n_kv * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(H, self.n_kv * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, H, bias=True)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.group_size = config.group_size
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta, config.rope_scaling)
        # Optional sink token (off by default for JSON parity)
        self.enable_sink = bool(config.enable_sink_token)
        if self.enable_sink:
            self.k_sink = nn.Parameter(torch.zeros(self.n_head, self.head_dim))
            self.v_sink = nn.Parameter(torch.zeros(self.n_head, self.head_dim))
            nn.init.normal_(self.k_sink, std=config.initializer_range)
            nn.init.normal_(self.v_sink, std=config.initializer_range)
            self.sink_logit_bias = float(config.sink_logit_bias)

        # Init
        for m in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.normal_(m.weight, mean=0.0, std=config.initializer_range)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _kv_expand(self, kv: torch.Tensor) -> torch.Tensor:
        # kv: (B, T, n_kv*Dh) -> (B, T, n_head, Dh), by repeating kv heads within each group
        B, T, _ = kv.shape
        kv = kv.view(B, T, self.n_kv, self.head_dim)
        kv = kv.unsqueeze(3).expand(B, T, self.n_kv, self.group_size, self.head_dim)
        kv = kv.reshape(B, T, self.n_head, self.head_dim)
        return kv

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,  # (T, T) boolean mask where True=keep, False=mask
        is_sliding_layer: bool = False,
        sliding_window: int = 128,
    ) -> torch.Tensor:
        B, T, H = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self._kv_expand(self.k_proj(x))
        v = self._kv_expand(self.v_proj(x))

        # Apply RoPE to q,k per head
        q = self.rope.apply_rotary(q.view(B * self.n_head, T, self.head_dim), positions.repeat_interleave(self.n_head, 0))
        k = self.rope.apply_rotary(k.view(B * self.n_head, T, self.head_dim), positions.repeat_interleave(self.n_head, 0))
        q = q.view(B, self.n_head, T, self.head_dim)
        k = k.view(B, self.n_head, T, self.head_dim)
        v = v.view(B, self.n_head, T, self.head_dim)

        # Optionally append sink token as an extra key/value (T_sink = 1)
        if self.enable_sink:
            k_sink = self.k_sink.view(1, self.n_head, 1, self.head_dim).expand(B, -1, -1, -1)
            v_sink = self.v_sink.view(1, self.n_head, 1, self.head_dim).expand(B, -1, -1, -1)
            k = torch.cat([k, k_sink], dim=2)  # (B, n_head, T+1, Dh)
            v = torch.cat([v, v_sink], dim=2)

        # Compute attention scores
        att = torch.einsum("bhtd,bhsd->bhts", q, k) * self.scale  # (B, n_head, T, S)
        S = att.size(-1)
        # Add bias to the sink column only
        if self.enable_sink:
            att[:, :, :, -1] = att[:, :, :, -1] + self.sink_logit_bias

        # Build masks
        if attn_mask is not None:
            # attn_mask: (T, T) -> broadcast to (1,1,T,S)
            if attn_mask.size(-1) != S:
                # Extend mask with a True column for the sink position if present
                if self.enable_sink:
                    sink_col = torch.ones(attn_mask.size(0), 1, dtype=attn_mask.dtype, device=attn_mask.device)
                    attn_mask_ext = torch.cat([attn_mask, sink_col], dim=-1)
                else:
                    raise RuntimeError("attn_mask last dim must match key length S")
            else:
                attn_mask_ext = attn_mask
            mask = attn_mask_ext.view(1, 1, T, S)
            att = att.masked_fill(~mask, float("-inf"))

        if is_sliding_layer:
            # Enforce a local window over source positions (ignoring the sink column)
            S_main = S - (1 if self.enable_sink else 0)
            idx = torch.arange(S_main, device=x.device)
            local = (idx.view(1, 1, 1, S_main) >= (idx.view(1, 1, S_main, 1) - sliding_window))
            local = local & (idx.view(1, 1, 1, S_main) <= idx.view(1, 1, S_main, 1))  # causal
            if self.enable_sink:
                # concat a True column for the sink (always allowed)
                sink_col = torch.ones(B, self.n_head, T, 1, dtype=torch.bool, device=x.device)
                local = torch.cat([local.expand(B, self.n_head, -1, -1), sink_col], dim=-1)
            else:
                local = local.expand(B, self.n_head, -1, -1)
            att = att.masked_fill(~local, float("-inf"))

        p = F.softmax(att, dim=-1)
        p = self.attn_drop(p)
        y = torch.einsum("bhts,bhsd->bhtd", p, v).contiguous()
        y = y.view(B, T, self.n_head * self.head_dim)
        y = self.o_proj(y)
        y = self.resid_drop(y)
        return y


class MoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        H = config.hidden_size
        E = config.num_local_experts
        FF = config.intermediate_size
        self.E = E
        self.K = int(config.experts_per_token)
        self.ffn_in = nn.Parameter(torch.empty(E, H, 2 * FF))
        self.ffn_in_bias = nn.Parameter(torch.zeros(E, 2 * FF))
        self.ffn_out = nn.Parameter(torch.empty(E, FF, H))
        self.ffn_out_bias = nn.Parameter(torch.zeros(E, H))
        self.router = nn.Linear(H, E, bias=True)
        self.swiglu_limit = float(config.swiglu_limit)
        self.router_aux_loss_coef = float(config.router_aux_loss_coef)

        for p in (self.ffn_in, self.ffn_out):
            nn.init.normal_(p, mean=0.0, std=config.initializer_range)
        nn.init.zeros_(self.ffn_in_bias)
        nn.init.zeros_(self.ffn_out_bias)
        nn.init.normal_(self.router.weight, mean=0.0, std=config.initializer_range)
        nn.init.zeros_(self.router.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: (B, T, H)
        B, T, H = x.shape
        logits = self.router(x)  # (B, T, E)
        probs = F.softmax(logits, dim=-1)
        # Load-balancing aux (Switch-style): encourage equal importance & equal load
        # importance: mean prob per expert; load: fraction of times picked as top-1
        importance = probs.mean(dim=(0, 1))  # (E,)
        top1 = probs.argmax(dim=-1)  # (B, T)
        load = F.one_hot(top1, num_classes=self.E).float().mean(dim=(0, 1))  # (E,)
        aux_loss = self.E * (importance * load).sum()

        # Top-K selection per token
        topk = torch.topk(probs, k=self.K, dim=-1, sorted=True)
        idx = topk.indices  # (B, T, K)
        wts = topk.values   # (B, T, K)
        wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-9)  # re-normalize

        # Gather expert params: (B, T, K, H, 2*FF)
        W1 = self.ffn_in[idx]          # (B,T,K,H,2F)
        b1 = self.ffn_in_bias[idx]     # (B,T,K,2F)
        U = torch.einsum("btkhf,bth->btkf", W1, x) + b1  # (B,T,K,2F)
        U = swiglu(U, limit=self.swiglu_limit)            # (B,T,K,F)

        W2 = self.ffn_out[idx]         # (B,T,K,F,H)
        b2 = self.ffn_out_bias[idx]    # (B,T,K,H)
        Z = torch.einsum("btkfh,btkf->btkh", W2, U) + b2  # (B,T,K,H)

        # Combine experts
        out = torch.einsum("btkh,btk->bth", Z, wts)       # (B,T,H)
        return out, {"router_aux_loss": aux_loss}


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MultiheadSelfAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = MoE(config)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_sliding_layer: bool,
        sliding_window: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        a = self.attn(self.norm1(x), positions, attn_mask, is_sliding_layer, sliding_window)
        x = x + a
        m, aux = self.moe(self.norm2(x))
        x = x + m
        return x, aux


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        H = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, H)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm_f = RMSNorm(H, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(H, config.vocab_size, bias=False)

        # tie weights if requested
        if not config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.initializer_range)
        else:
            self.lm_head.weight = self.embed_tokens.weight

        # Embedding init
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=config.initializer_range)

    @staticmethod
    def build_attn_mask(T: int, device, dtype=torch.bool) -> torch.Tensor:
        # Standard causal mask: allow s<=t
        i = torch.arange(T, device=device)
        j = torch.arange(T, device=device)
        mask = (j[None, :] <= i[:, None])
        return mask.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,           # (B, T)
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T = input_ids.shape
        device = input_ids.device
        x = self.embed_tokens(input_ids)
        x = self.drop(x)
        positions = torch.arange(T, device=device).view(1, T).expand(B, -1)

        attn_mask = self.build_attn_mask(T, device)  # (T, T)

        aux_losses: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            is_sliding = (self.config.layer_types[i] == "sliding_attention")
            x, aux = layer(x, positions, attn_mask, is_sliding, self.config.sliding_window)
            if aux and "router_aux_loss" in aux:
                aux_losses.append(aux["router_aux_loss"])

        x = self.norm_f(x)
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        aux_out: Dict[str, torch.Tensor] = {}
        if labels is not None:
            # Next-token prediction (shifted)
            logits_flat = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(logits_flat, targets, ignore_index=-100)
            if aux_losses:
                aux_total = torch.stack(aux_losses).mean()
                aux_weighted = self.config.router_aux_loss_coef * aux_total
                loss = loss + aux_weighted
                aux_out["router_aux_loss"] = aux_total.detach()
                aux_out["router_aux_loss_weighted"] = aux_weighted.detach()
        return logits, {"loss": loss, **aux_out}

    @classmethod
    def from_hf_config(cls, hf_config: Dict[str, Any]) -> "Transformer":
        cfg = ModelConfig.from_hf_dict(hf_config)
        return cls(cfg)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,  # (B, T)
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        eos_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        B = input_ids.size(0)
        tokens = input_ids
        for _ in range(max_new_tokens):
            logits, _ = self(tokens)
            next_logits = logits[:, -1, :]
            if temperature != 1.0:
                next_logits = next_logits / max(1e-6, temperature)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if eos_id is not None:
                # stop if all sequences have ended
                if (next_token.squeeze(-1) == eos_id).all():
                    break
        return tokens

@dataclass
class GPTConfig:
    """
    Configuration class for the gpt-oss-120b model. This holds the architectural
    hyperparameters.
    """
    # Core architecture parameters
    block_size: int = 131072
    vocab_size: int = 201088
    n_layer: int = 36
    n_embd: int = 2880

    # Grouped-Query Attention (GQA) parameters
    n_head: int = 64
    n_kv_head: int = 8
    head_dim: int = 64

    # Mixture-of-Experts (MoE) parameters
    num_experts: int = 128
    experts_per_token: int = 4

    # Feed-forward network (within each expert) configuration
    intermediate_size: int = 2880

    # Advanced parameters (kept for completeness)
    attention_bias: bool = True
    tie_word_embeddings: bool = False # From model card

def calculate_params_from_config(config: GPTConfig) -> dict:
    """
    Calculates the theoretical parameter count from a config dataclass
    without instantiating the model.
    """
    # Embedding & Unembedding
    embed_params = config.vocab_size * config.n_embd
    unembed_params = 0 if config.tie_word_embeddings else config.n_embd * config.vocab_size

    # Attention per layer
    attn_params_per_layer = (
        (config.n_embd * (config.n_head * config.head_dim)) + ((config.n_head * config.head_dim) if config.attention_bias else 0) + # Q
        (config.n_embd * (config.n_kv_head * config.head_dim)) + ((config.n_kv_head * config.head_dim) if config.attention_bias else 0) + # K
        (config.n_embd * (config.n_kv_head * config.head_dim)) + ((config.n_kv_head * config.head_dim) if config.attention_bias else 0) + # V
        ((config.n_head * config.head_dim) * config.n_embd) + (config.n_embd if config.attention_bias else 0)   # O
    )

    # Router per layer
    gate_params_per_layer = config.n_embd * config.num_experts + config.num_experts

    # Norms
    norm_params = (config.n_layer * 2 + 1) * config.n_embd

    # Single Expert parameters (with biases)
    expert_params = (
        (config.n_embd * config.intermediate_size * 2) + # ffn_in weight
        (config.intermediate_size * 2) +                      # ffn_in bias
        (config.intermediate_size * config.n_embd) +   # ffn_out weight
        (config.n_embd)                                  # ffn_out bias
    )

    # --- Aggregate ---
    total_expert_params = config.n_layer * config.num_experts * expert_params
    shared_params = (
        embed_params + unembed_params +
        config.n_layer * (attn_params_per_layer + gate_params_per_layer) +
        norm_params
    )
    total_params = total_expert_params + shared_params

    # Active parameters
    active_expert_params = config.n_layer * config.experts_per_token * expert_params
    # Model card counts unembedding as active, but not embedding.
    active_shared_params = shared_params - embed_params   # keep lm_head, drop embeddings
    active_params = active_shared_params + active_expert_params

    return {
        "total_B": total_params / 1e9,
        "active_B": active_params / 1e9,
        "expert_B": total_expert_params / 1e9,
        "shared_B": shared_params / 1e9,
    }


if __name__ == "__main__":
    # 4. Now, run the calculation
    config_120b = GPTConfig()
    params = calculate_params_from_config(config_120b)
    print(f"Total Theoretical Parameters: {params['total_B']:.2f}B")
    print(f"Active Theoretical Parameters: {params['active_B']:.2f}B")
    print(f"  - Total Expert Parameters: {params['expert_B']:.2f}B")
    print(f"  - Total Shared Parameters: {params['shared_B']:.2f}B")