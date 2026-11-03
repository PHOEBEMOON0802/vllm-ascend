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
    row_chunk: int = 64,
) -> torch.Tensor:
    """Batched CE+TV gradient for vanilla_residual head mode.

    Computes dL/d(delta) for each request in the batch, where:
        q_logits = base + (e + e @ delta) @ W2.T
        L = ce_weight * L_ce + tv_weight * L_tv + anchor_lambda * L_anchor

    Fused implementation: each of q / teacher / ref gets exactly ONE
    softmax over the vocab dim, reused for both the W2 projection and the
    TV indicator (3 full-vocab passes instead of 5), and the per-request
    bias bmm is flattened into one [B*K, r] @ [r, V] matmul. Rows are
    processed in ``row_chunk`` chunks to bound the peak extra memory to
    ~3 * row_chunk * V floats (B=1 workloads fit in a single chunk, which
    minimizes kernel-launch count on NPU).

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
        row_chunk: rows (B*K) per fused softmax/projection chunk.

    Returns:
        [B, r, r] gradient tensor per request.
    """
    e = e_batch[:, :K_actual, :].to(torch.float32)          # [B, K, r]
    base = base_batch[:, :K_actual, :].to(torch.float32)    # [B, K, V]
    deltas = deltas_batch.to(torch.float32)                 # [B, r, r]
    W2_f = W2.to(torch.float32)                             # [V, r]
    tl = tl_batch[:, :K_actual, :].to(torch.float32)        # [B, K, V]
    alpha = alpha_batch[:, :K_actual].to(torch.float32)     # [B, K]

    B, K, r = e.shape
    rows = B * K

    # q_logits = base + (e + e @ delta) @ W2.T ; ref = base + e @ W2.T
    e_delta = torch.bmm(e, deltas)                          # [B, K, r]
    base_flat = base.reshape(rows, -1)                      # [rows, V]
    q_logits = base_flat + (e + e_delta).reshape(rows, r) @ W2_f.t()
    ref_logits = base_flat + e.reshape(rows, r) @ W2_f.t()
    tl_flat = tl.reshape(rows, -1)                          # [rows, V]

    # du = dL/d(q_proj), computed in row chunks with softmax reuse.
    du_flat = torch.empty(rows, r, dtype=torch.float32, device=e.device)
    for start in range(0, rows, row_chunk):
        end = min(start + row_chunk, rows)
        q_sm = torch.softmax(q_logits[start:end], dim=-1)
        p_sm = torch.softmax(tl_flat[start:end], dim=-1)
        ref_sm = torch.softmax(ref_logits[start:end], dim=-1)
        q_proj = q_sm @ W2_f                                # [c, r]
        teacher_proj = p_sm @ W2_f
        ref_proj = ref_sm @ W2_f
        # CE: ce_weight * (q_proj - teacher_proj)
        du = ce_weight * (q_proj - teacher_proj)
        # TV: indicator = (q_softmax < p_softmax), reusing the softmaxes.
        indicator_weighted = (q_sm < p_sm).to(torch.float32) * q_sm
        du = du + tv_weight * (
            indicator_weighted @ W2_f
            - indicator_weighted.sum(-1, keepdim=True) * q_proj
        )
        # Anchor: anchor_lambda * (q_proj - ref_proj)
        du = du + anchor_lambda * (q_proj - ref_proj)
        du_flat[start:end] = du

    # Apply position weights and project to delta: grad = e.T @ (du * alpha)
    du = du_flat.view(B, K, r) * alpha.unsqueeze(-1)        # [B, K, r]
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
    row_chunk: int = 64,
) -> torch.Tensor:
    """Batched KL divergence gradient for vanilla_residual head mode.

    L = sum_k alpha_k * (KL(p_T || q) + anchor_lambda * KL(q_ref || q))

    Fused like the CE+TV path: one softmax per distribution per row chunk,
    flattened projection matmuls.
    """
    e = e_batch[:, :K_actual, :].to(torch.float32)
    base = base_batch[:, :K_actual, :].to(torch.float32)
    deltas = deltas_batch.to(torch.float32)
    W2_f = W2.to(torch.float32)
    tl = tl_batch[:, :K_actual, :].to(torch.float32)
    alpha = alpha_batch[:, :K_actual].to(torch.float32)

    B, K, r = e.shape
    rows = B * K

    e_delta = torch.bmm(e, deltas)
    base_flat = base.reshape(rows, -1)
    q_logits = base_flat + (e + e_delta).reshape(rows, r) @ W2_f.t()
    ref_logits = base_flat + e.reshape(rows, r) @ W2_f.t()
    tl_flat = tl.reshape(rows, -1)

    du_flat = torch.empty(rows, r, dtype=torch.float32, device=e.device)
    for start in range(0, rows, row_chunk):
        end = min(start + row_chunk, rows)
        q_proj = torch.softmax(q_logits[start:end], dim=-1) @ W2_f
        teacher_proj = torch.softmax(tl_flat[start:end], dim=-1) @ W2_f
        ref_proj = torch.softmax(ref_logits[start:end], dim=-1) @ W2_f
        # KL(teacher || q): (q_proj - teacher_proj);
        # KL(ref || q): anchor_lambda * (q_proj - ref_proj)
        du_flat[start:end] = (
            (q_proj - teacher_proj) + anchor_lambda * (q_proj - ref_proj)
        )

    du = du_flat.view(B, K, r) * alpha.unsqueeze(-1)
    # Forward applies e @ delta, so dL/d(delta) = e.T @ du.
    delta_grad = torch.einsum('bki,bkj->bij', e, du)
    return delta_grad


@torch.no_grad()
def position_weights(k: int, gamma: float, *, device, dtype) -> torch.Tensor:
    """Exponential decay position weights: alpha[k] = exp(-k / gamma)."""
    w = torch.exp(-torch.arange(k, device=device, dtype=dtype) / gamma)
    return w