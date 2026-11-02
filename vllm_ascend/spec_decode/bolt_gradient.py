# SPDX-License-Identifier: Apache-2.0
"""BOLT gradient computation functions (analytical, no autograd).

All gradients are computed analytically for efficiency. The core operations are:
  - ``exact_projection``: softmax(logits) @ W2, computed in chunks to avoid
    materializing the full [V, V] matrix.
  - ``cetv_residual_gradient_batch``: CE+TV loss gradient w.r.t. delta for
    the ``vanilla_residual`` head mode (delta shape [r, r]).

The gradient formulas are derived as follows:

For vanilla_residual mode, the draft logits are:
    q_logits = base + (markov_emb + markov_emb @ delta) @ W2.T

Let proj(q) = softmax(q_logits) @ W2. The CE+TV loss is:
    L_ce = 0.5 * ||proj(q) - proj(teacher)||^2
    L_tv = 0.5 * sum(indicator * (q_softmax - p_softmax) * proj(q))
    L_anchor = anchor_lambda * 0.5 * ||proj(q) - proj(ref)||^2

where ref = base + markov_emb @ W2.T (delta=0).

The gradient dL/d(delta) is computed via chain rule:
    dL/d(proj(q)) = du
    d(proj(q))/d(delta) = markov_emb.T @ (diag(q_softmax) - q_softmax @ q_softmax.T) @ W2

Simplified (see ARCHITECTURE.md for derivation):
    delta_grad = e.T @ (du * alpha)  →  [r, r]
"""
from __future__ import annotations

import torch


@torch.no_grad()
def exact_projection(logits: torch.Tensor, W2: torch.Tensor,
                     chunk_size: int = 32768) -> torch.Tensor:
    """Compute softmax(logits) @ W2 in chunks.

    Args:
        logits: [..., V] logits tensor.
        W2: [V, r] projection weight.
        chunk_size: number of vocab tokens per chunk (memory control).

    Returns:
        [..., r] projected tensor.
    """
    x = logits.to(torch.float32)
    proj = W2.to(torch.float32)
    # Numerically stable softmax: subtract max before exp.
    maximum = x.amax(-1, keepdim=True)
    denominator = x.new_zeros(x.shape[:-1] + (1,))
    numerator = x.new_zeros(x.shape[:-1] + (W2.shape[1],))
    for start in range(0, x.shape[-1], chunk_size):
        end = start + chunk_size
        weights = (x[..., start:end] - maximum).exp()
        denominator.add_(weights.sum(-1, keepdim=True))
        numerator.add_(weights @ proj[start:end])
    return numerator / denominator


@torch.no_grad()
def cetv_residual_gradient_batch(
    e_batch: torch.Tensor,          # [B, K, r] markov embeddings
    base_batch: torch.Tensor,       # [B, K, V] base logits
    deltas_batch: torch.Tensor,     # [B, r, r] per-request deltas
    W2: torch.Tensor,               # [V, r] frozen projection weight
    tl_batch: torch.Tensor,         # [B, K, V] teacher logits
    alpha_batch: torch.Tensor,      # [B, K] position weights
    anchor_lambda: float,           # anchor regularization coefficient
    K_actual: int,                  # number of draft positions
    ce_weight: float = 0.5,
    tv_weight: float = 0.5,
) -> torch.Tensor:
    """Batched CE+TV gradient for vanilla_residual head mode.

    Computes dL/d(delta) for each request in the batch, where:
        q_logits = base + (e + e @ delta) @ W2.T
        L = ce_weight * L_ce + tv_weight * L_tv + anchor_lambda * L_anchor

    Args:
        e_batch: [B, K, r] markov embeddings (W1[prev_token]).
        base_batch: [B, K, V] base logits from draft model.
        deltas_batch: [B, r, r] per-request delta matrices.
        W2: [V, r] frozen markov_w2 weight.
        tl_batch: [B, K, V] teacher (target model) logits.
        alpha_batch: [B, K] position decay weights.
        anchor_lambda: anchor regularization coefficient.
        K_actual: number of draft positions to use (<=K).
        ce_weight: CE loss weight.
        tv_weight: TV loss weight.

    Returns:
        [B, r, r] gradient tensor per request.
    """
    e = e_batch[:, :K_actual, :].to(torch.float32)          # [B, K, r]
    base = base_batch[:, :K_actual, :].to(torch.float32)    # [B, K, V]
    deltas = deltas_batch.to(torch.float32)           # [B, r, r]
    W2_f = W2.to(torch.float32)                             # [V, r]
    tl = tl_batch[:, :K_actual, :].to(torch.float32)        # [B, K, V]
    alpha = alpha_batch[:, :K_actual].to(torch.float32)     # [B, K]

    B, K, r = e.shape

    # Compute q_logits = base + (e + e @ delta) @ W2.T
    # e @ delta: [B, K, r] @ [B, r, r] -> [B, K, r]
    e_delta = torch.bmm(e, deltas)                          # [B, K, r]
    # (e + e_delta) @ W2.T: [B, K, r] @ [r, V] -> [B, K, V]
    bias = torch.bmm(e + e_delta, W2_f.t().unsqueeze(0).expand(B, -1, -1))
    q_logits = base + bias                                  # [B, K, V]

    # Compute reference logits (delta=0): ref = base + e @ W2.T
    ref_bias = torch.bmm(e, W2_f.t().unsqueeze(0).expand(B, -1, -1))
    ref_logits = base + ref_bias                            # [B, K, V]

    # Projections: softmax(logits) @ W2
    q_proj = exact_projection(q_logits, W2_f)               # [B, K, r]
    teacher_proj = exact_projection(tl, W2_f)               # [B, K, r]
    ref_proj = exact_projection(ref_logits, W2_f)           # [B, K, r]

    # du = dL/d(q_proj)
    # CE: dL_ce/d(q_proj) = ce_weight * (q_proj - teacher_proj)
    du = ce_weight * (q_proj - teacher_proj)                # [B, K, r]

    # TV: dL_tv/d(q_proj) involves the indicator function.
    # indicator = (q_softmax < p_softmax) where p = softmax(teacher_logits)
    # The TV gradient simplifies to: tv_weight * indicator_weighted * q_proj
    # We computed it numerically via softmax probabilities.
    q_softmax = torch.softmax(q_logits, dim=-1)             # [B, K, V]
    p_softmax = torch.softmax(tl, dim=-1)                   # [B, K, V]
    indicator = (q_softmax < p_softmax).to(torch.float32)   # [B, K, V]

    # TV gradient: tv_weight * (indicator_proj - indicator_sum * q_proj)
    # where indicator_sum = (indicator * q_softmax) @ W2
    # and indicator_sum = (indicator * q_softmax).sum(-1, keepdim=True)
    indicator_weighted = indicator * q_softmax              # [B, K, V]
    indicator_proj = torch.bmm(indicator_weighted, W2_f.unsqueeze(0).expand(B, -1, -1))  # [B, K, r]
    indicator_sum = indicator_weighted.sum(-1, keepdim=True)  # [B, K, 1]
    du = du + tv_weight * (indicator_proj - indicator_sum * q_proj)  # [B, K, r]

    # Anchor: dL_anchor/d(q_proj) = anchor_lambda * (q_proj - ref_proj)
    du = du + anchor_lambda * (q_proj - ref_proj)           # [B, K, r]

    # Apply position weights
    du = du * alpha.unsqueeze(-1)                           # [B, K, r]

    # Project to delta: forward applies e @ delta, so
    # dL/d(delta)[i, j] = sum_k e[k, i] * du[k, j]  ->  e.T @ du
    delta_grad = torch.einsum('bki,bkj->bij', e, du)        # [B, r, r]

    return delta_grad


@torch.no_grad()
def kl_residual_gradient_batch(
    e_batch: torch.Tensor,          # [B, K, r]
    base_batch: torch.Tensor,       # [B, K, V]
    deltas_batch: torch.Tensor,     # [B, r, r]
    W2: torch.Tensor,               # [V, r]
    tl_batch: torch.Tensor,         # [B, K, V]
    alpha_batch: torch.Tensor,      # [B, K]
    anchor_lambda: float,
    K_actual: int,
) -> torch.Tensor:
    """Batched KL divergence gradient for vanilla_residual head mode.

    L = sum_k alpha_k * (KL(p_T || q) + anchor_lambda * KL(q_red || q))
    """
    e = e_batch[:, :K_actual, :].to(torch.float32)
    base = base_batch[:, :K_actual, :].to(torch.float32)
    deltas = deltas_batch.to(torch.float32)
    W2_f = W2.to(torch.float32)
    tl = tl_batch[:, :K_actual, :].to(torch.float32)
    alpha = alpha_batch[:, :K_actual].to(torch.float32)

    B, K, r = e.shape

    e_delta = torch.bmm(e, deltas)
    bias = torch.bmm(e + e_delta, W2_f.t().unsqueeze(0).expand(B, -1, -1))
    q_logits = base + bias

    ref_bias = torch.bmm(e, W2_f.t().unsqueeze(0).expand(B, -1, -1))
    ref_logits = base + ref_bias

    q_proj = exact_projection(q_logits, W2_f)
    teacher_proj = exact_projection(tl, W2_f)
    ref_proj = exact_projection(ref_logits, W2_f)

    # KL(teacher || q) gradient: (q_proj - teacher_proj)
    du = (q_proj - teacher_proj)

    # KL(ref || q) gradient: anchor_lambda * (q_proj - ref_proj)
    du = du + anchor_lambda * (q_proj - ref_proj)

    du = du * alpha.unsqueeze(-1)
    # Forward applies e @ delta, so dL/d(delta) = e.T @ du.
    delta_grad = torch.einsum('bki,bkj->bij', e, du)
    return delta_grad


@torch.no_grad()
def position_weights(k: int, gamma: float, *, device, dtype) -> torch.Tensor:
    """Exponential decay position weights: alpha[k] = exp(-k / gamma)."""
    w = torch.exp(-torch.arange(k, device=device, dtype=dtype) / gamma)
    return w