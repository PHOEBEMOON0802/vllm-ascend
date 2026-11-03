# SPDX-License-Identifier: Apache-2.0
"""BOLT (Block-level Online Learning for Target-aligned draft) Manager.

Integrates online adaptive speculative decoding into the vLLM-Ascend DSpark
proposer/verify chain. Each request gets its own delta matrix that is updated
after each verify block using the teacher (target model) logits, improving
draft token acceptance rates over the course of a single request.

Key design decisions:
  - Delta is request-private: ``states[req_id]`` holds per-request optimizer
    state (delta, momentum, step count, metrics).
  - Gradients are computed analytically (no autograd) for efficiency.
  - The frozen markov head weights (W1, W2) are never modified — BOLT computes
    an additive delta bias that is applied during the markov bias computation
    in ``llm_base_proposer._propose()``.
  - Two head modes supported:
      ``vanilla_residual``: delta [r, r], bias = (markov_emb + markov_emb @ delta) @ W2.T
      ``bounded``: delta [r, 2r+d], applied to RNN joint_proj output rows.

Flow: propose (save_proposal) → verify (record_verify) → update (bolt_update_batch).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from vllm.logger import logger

from vllm_ascend.spec_decode.bolt_gradient import (
    cetv_residual_gradient_batch,
    kl_residual_gradient_batch,
    position_weights,
)

if TYPE_CHECKING:
    from vllm_ascend.ascend_config import BoltConfig


def _env_int(name: str, default: int) -> int:
    """Parse an int env var, falling back to default on absence/garbage."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Optimizers (per-request state)
# ---------------------------------------------------------------------------

class RequestMuon:
    """Muon optimizer: Newton-Schulz orthogonalization on momentum buffer.

    Uses Newton-Schulz orthogonalization (5 iterations) with Nesterov momentum.
    Reference: https://github.com/KellerJordan/Muon

    Interface: ``.delta``, ``.step(gradient) -> (gnorm, clipped)``, ``.version``
    """

    def __init__(self, shape, device, lr=0.005, momentum=0.95, clip=1.0):
        self.lr = lr
        self.momentum = momentum
        self.clip = clip
        self.delta = torch.zeros(shape, device=device, dtype=torch.float32)
        self.buf = torch.zeros_like(self.delta)     # momentum buffer
        self.step_count = 0
        self.version = 0

    @torch.no_grad()
    def step(self, gradient: torch.Tensor) -> tuple[float, float]:
        g = gradient.float()
        gnorm = torch.linalg.vector_norm(g).item()
        factor = self.clip / (gnorm + 1e-12)
        clipped = 1.0 if factor < 1.0 else 0.0
        factor = min(factor, 1.0)
        g = g * factor
        self.step_count += 1

        # Nesterov momentum: buf.lerp_(g, 1-mu); update = g.lerp(buf, mu)
        mu = self.momentum
        self.buf.lerp_(g, 1 - mu)
        update = g.lerp(self.buf, mu)

        # Newton-Schulz orthogonalization (5 iterations)
        X = update.bfloat16()
        X.div_(X.norm().clamp(min=1e-7))
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(5):
            gram = X @ X.t()
            gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
            X = torch.addmm(X, gram_update, X, beta=a)
        self.delta.add_(X.float(), alpha=-self.lr)
        self.version += 1
        return gnorm, clipped


class RequestSGD:
    """SGD with optional momentum and gradient clipping.

    Interface: ``.delta``, ``.step(gradient) -> (gnorm, clipped)``, ``.version``
    """

    def __init__(self, shape, device, lr=0.005, momentum=0.0, clip=1.0):
        self.lr = lr
        self.momentum = momentum
        self.clip = clip
        self.delta = torch.zeros(shape, device=device, dtype=torch.float32)
        self.m = torch.zeros_like(self.delta) if momentum > 0 else None
        self.step_count = 0
        self.version = 0

    @torch.no_grad()
    def step(self, gradient: torch.Tensor) -> tuple[float, float]:
        g = gradient.float()
        gnorm = torch.linalg.vector_norm(g).item()
        factor = self.clip / (gnorm + 1e-12)
        clipped = 1.0 if factor < 1.0 else 0.0
        factor = min(factor, 1.0)
        g = g * factor
        self.step_count += 1
        if self.momentum > 0:
            self.m.mul_(self.momentum).add_(g)
            g = self.m
        self.delta.add_(g, alpha=-self.lr)
        self.version += 1
        return gnorm, clipped


class RequestAdam:
    """Standard Adam optimizer for per-request delta.

    Interface: ``.delta``, ``.step(gradient) -> (gnorm, clipped)``, ``.version``
    """

    def __init__(self, shape, device, lr=0.005, beta1=0.9, beta2=0.999,
                 eps=1e-8, clip=1.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.clip = clip
        self.delta = torch.zeros(shape, device=device, dtype=torch.float32)
        self.m = torch.zeros_like(self.delta)
        self.v = torch.zeros_like(self.delta)
        self.step_count = 0
        self.version = 0

    @torch.no_grad()
    def step(self, gradient: torch.Tensor) -> tuple[float, float]:
        g = gradient.float()
        gnorm = torch.linalg.vector_norm(g).item()
        factor = self.clip / (gnorm + 1e-12)
        clipped = 1.0 if factor < 1.0 else 0.0
        factor = min(factor, 1.0)
        g = g * factor
        self.step_count += 1
        self.m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
        self.v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
        mh = self.m / (1 - self.beta1 ** self.step_count)
        vh = self.v / (1 - self.beta2 ** self.step_count)
        self.delta.addcdiv_(mh, vh.sqrt() + self.eps, value=-self.lr)
        self.version += 1
        return gnorm, clipped


# ---------------------------------------------------------------------------
# Per-request state
# ---------------------------------------------------------------------------

@dataclass
class ReqState:
    """Per-request BOLT state."""
    optimizer: RequestMuon | RequestSGD | RequestAdam | None = None
    update_count: int = 0
    verify_rounds: int = 0
    accepted_draft_tokens: int = 0
    committed_tokens: int = 0
    zero_accept_rounds: int = 0
    full_block_rounds: int = 0
    last_accepted: int = 0
    # Per-round acceptance histogram: accept_counts[k] = # rounds with k accepted
    accept_hist: list = field(default_factory=lambda: [0] * 8)
    # Gradient statistics
    grad_norm_sum: float = 0.0
    grad_norm_max: float = 0.0
    clip_ratio_sum: float = 0.0
    # Saved proposal outputs (set during propose, consumed during update)
    saved_e: torch.Tensor | None = None       # [1, K, r] markov embeddings
    saved_base: torch.Tensor | None = None    # [1, K, V] base logits
    saved_draft: torch.Tensor | None = None   # [1, K] draft token ids
    # Propose round that produced the saved tensors (stale-pairing guard).
    saved_round: int = -1
    finished: bool = False


# ---------------------------------------------------------------------------
# BoltManager
# ---------------------------------------------------------------------------

class BoltManager:
    """Request-isolated BOLT state manager.

    Each request gets its own ``ReqState`` (delta + optimizer state + metrics).
    The frozen markov head weights (W1, W2) are never modified — BOLT computes
    an additive delta that is applied during the markov bias computation.

    Lifecycle:
      1. ``on_request_new(req_id)`` — create fresh delta = zeros
      2. ``save_proposal(req_id, e, base, draft_tokens)`` — called during propose
      3. ``record_verify(req_id, target_logits, spec_decode_metadata)`` — after verify
      4. ``bolt_update_batch(req_ids, target_logits_list)`` — compute gradient + update
      5. ``on_request_finished(req_id)`` — output metrics + release state

    During propose, ``get_delta(req_id)`` returns the current delta to be applied
    to the markov bias.
    """

    def __init__(
        self,
        config: BoltConfig,
        r: int = 256,
        d: int = 2560,
        K: int = 7,
        device: str | torch.device = "npu",
    ):
        self.config = config
        self.r = r
        self.d = d
        self.K = K
        self.device = torch.device(device) if isinstance(device, str) else device
        self.anchor_lambda = config.anchor_lambda

        # Delta shape depends on head mode
        if config.head_mode == "vanilla_residual":
            self._delta_shape = (r, r)
        else:
            # bounded (RNN output-only): delta [r, 2r+d]
            self._delta_shape = (r, 2 * r + d)

        # Position weights (precomputed)
        self.alpha = position_weights(
            K, config.gamma, device=self.device, dtype=torch.float32
        )  # [K]

        # Per-request state dict: req_id → ReqState
        self.states: dict[str, ReqState] = {}

        # Weight references (set via set_weights, frozen)
        self.W1: torch.Tensor | None = None   # [V, r] markov_w1 weight
        self.W2: torch.Tensor | None = None   # [V, r] markov_w2 weight
        # RNN-only weights (for bounded mode)
        self.W0: torch.Tensor | None = None   # [3r, 2r+d] joint_proj weight
        self.b0: torch.Tensor | None = None   # [3r] joint_proj bias

        self.enabled = config.enabled

        # Propose-round counter. Advanced by the model runner once per real
        # execute step (begin_round); saved proposals from older rounds were
        # never verified and must not be paired with later target logits.
        self.propose_round = 0

        # Metrics
        self._metrics_dir = config.metrics_dir
        self._finish_counter = 0
        # One-shot warning flags (avoid log spam on hot paths)
        self._verify_error_logged = False
        self._vocab_mismatch_logged = False

        # Debug printing (observe per-request metric evolution live).
        # Controlled by env vars, in addition to bolt_config.debug:
        #   VLLM_ASCEND_BOLT_DEBUG: 0=off (default), 1=per-verify-block
        #     acceptance line + per-request finish summary, 2=additionally
        #     per-update gradient/delta statistics.
        #   VLLM_ASCEND_BOLT_DEBUG_INTERVAL: print the verify line every N
        #     blocks per request (default 1; raise to tame high-QPS logs).
        # bolt_config.debug=true is equivalent to level 2.
        self._debug_level = _env_int("VLLM_ASCEND_BOLT_DEBUG", 0)
        if config.debug and self._debug_level < 2:
            self._debug_level = 2
        self._debug_interval = max(
            1, _env_int("VLLM_ASCEND_BOLT_DEBUG_INTERVAL", 1)
        )
        # Update-timing accumulators (debug level >= 1 only)
        self._update_time_sum_ms = 0.0
        self._update_time_count = 0
        self._update_time_max_ms = 0.0
        if self._debug_level > 0:
            logger.info(
                "[BOLT] debug printing enabled: level=%d interval=%d "
                "(VLLM_ASCEND_BOLT_DEBUG / VLLM_ASCEND_BOLT_DEBUG_INTERVAL)",
                self._debug_level, self._debug_interval,
            )

    @property
    def metrics_enabled(self) -> bool:
        """Whether per-request metrics collection is requested."""
        return bool(self._metrics_dir)

    # ---- Weight management ----

    def set_weights(
        self,
        W1: torch.Tensor,
        W2: torch.Tensor,
        W0: torch.Tensor | None = None,
        b0: torch.Tensor | None = None,
    ):
        """Set references to draft model weights (frozen, never modified).

        Args:
            W1: [V, r] markov_w1 weight (embedding table).
            W2: [V, r] markov_w2 weight (projection).
            W0: [3r, 2r+d] joint_proj weight (RNN mode only, optional).
            b0: [3r] joint_proj bias (RNN mode only, optional).
        """
        self.W1 = W1
        self.W2 = W2
        self.W0 = W0
        self.b0 = b0

    # ---- Request lifecycle ----

    def on_request_new(self, req_id: str):
        """Create fresh state for a new request."""
        st = ReqState()
        if self.config.optimizer == "muon":
            st.optimizer = RequestMuon(
                self._delta_shape, self.device,
                lr=self.config.lr, momentum=0.95, clip=self.config.clip,
            )
        elif self.config.optimizer == "sgdm":
            st.optimizer = RequestSGD(
                self._delta_shape, self.device,
                lr=self.config.lr, momentum=self.config.sgd_momentum,
                clip=self.config.clip,
            )
        else:  # adam
            st.optimizer = RequestAdam(
                self._delta_shape, self.device,
                lr=self.config.lr, clip=self.config.clip,
            )
        self.states[req_id] = st

    def on_request_finished(self, req_id: str):
        """Output per-request statistics and release state."""
        st = self.states.pop(req_id, None)
        if st is None:
            return
        st.finished = True
        self._finish_counter += 1
        self._output_request_metrics(req_id, st, self._finish_counter)
        if self._debug_level >= 1:
            rounds = max(1, st.verify_rounds)
            delta_max = (
                st.optimizer.delta.abs().max().item()
                if st.optimizer is not None and st.update_count > 0 else 0.0
            )
            logger.info(
                "[BOLT][finish] req=%s rounds=%d accepted_total=%d "
                "tau=%.3f tau_bonus=%.3f committed=%d updates=%d "
                "delta_max=%.6f",
                req_id, st.verify_rounds, st.accepted_draft_tokens,
                st.accepted_draft_tokens / rounds,
                (st.accepted_draft_tokens + st.verify_rounds) / rounds,
                st.committed_tokens, st.update_count, delta_max,
            )

    def _output_request_metrics(self, req_id: str, st: ReqState, finish_idx: int):
        if not self._metrics_dir:
            return
        try:
            os.makedirs(self._metrics_dir, exist_ok=True)
            metrics = self._compute_metrics(st)
            metrics["finish_idx"] = finish_idx
            metrics["req_id"] = req_id
            # Request ids may embed user-supplied strings; sanitize so the
            # filename cannot escape metrics_dir.
            safe_req_id = re.sub(r"[^\w.-]", "_", req_id)
            out_path = os.path.join(
                self._metrics_dir, f"metrics_{safe_req_id}.json"
            )
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(metrics, f, indent=2)
            os.replace(tmp_path, out_path)
        except Exception:
            logger.warning(
                "BOLT: failed to write metrics for request %s",
                req_id, exc_info=True,
            )

    def _compute_metrics(self, st: ReqState) -> dict:
        rounds = max(1, st.verify_rounds)
        return {
            "verify_rounds": st.verify_rounds,
            "accepted_draft_tokens": st.accepted_draft_tokens,
            "committed_tokens": st.committed_tokens,
            "zero_accept_rounds": st.zero_accept_rounds,
            "full_block_rounds": st.full_block_rounds,
            "tau_no_bonus": st.accepted_draft_tokens / rounds,
            "tau_with_bonus": (st.accepted_draft_tokens + rounds) / rounds,
            "j0": st.zero_accept_rounds / rounds,
            "full_block_rate": st.full_block_rounds / rounds,
            "update_count": st.update_count,
            "delta_max": (
                st.optimizer.delta.abs().max().item()
                if st.update_count > 0 else 0.0
            ),
            "grad_norm_mean": st.grad_norm_sum / max(1, st.update_count),
            "grad_norm_max": st.grad_norm_max,
            "clip_ratio": st.clip_ratio_sum / max(1, st.update_count),
            "accept_hist": st.accept_hist[:8],
            "p_a_ge_k": [
                sum(st.accept_hist[k:]) / max(1, rounds) for k in range(8)
            ],
        }

    # ---- Delta access ----

    def get_delta(self, req_id: str) -> torch.Tensor | None:
        """Return the current delta for a request.

        Returns None if BOLT is disabled or the request has no state.
        """
        if not self.enabled:
            return None
        st = self.states.get(req_id)
        if st is None or st.optimizer is None:
            return None
        return st.optimizer.delta.to(
            self.W2.dtype if self.W2 is not None else torch.float32
        )

    # ---- Proposal (save data for later update) ----

    def begin_round(self):
        """Start a new propose round.

        Called by the model runner once per real execute step, after the
        update for the previous round and before the next propose. Saved
        proposals from older rounds are stale (their drafts were never
        verified against target logits) and are released here.
        """
        self.propose_round += 1
        for st in self.states.values():
            st.saved_e = None
            st.saved_base = None
            st.saved_draft = None

    def save_proposal(
        self,
        req_id: str,
        e: torch.Tensor | None,       # [1, K, r] markov embeddings
        base: torch.Tensor | None,    # [1, K, V] base logits (pre markov-bias)
        draft_tokens: torch.Tensor,   # [1, K] draft token ids
    ):
        """Save draft model outputs for later BOLT update.

        Called during propose, after the markov bias loop completes.
        ``e``/``base`` may be None in metrics-only mode (verify metrics need
        only the draft tokens). All tensors are cloned: the caller's buffers
        are reused across propose cycles and must not be aliased.
        Requests without state (lifecycle not started by the runner) are
        ignored so that state creation stays owned by ``on_request_new``.
        """
        st = self.states.get(req_id)
        if st is None:
            return
        st.saved_e = e.detach().clone() if e is not None else None
        st.saved_base = base.detach().clone() if base is not None else None
        st.saved_draft = draft_tokens.detach().clone()
        st.saved_round = self.propose_round

    # ---- Verification metrics ----

    def record_verify(
        self,
        req_id: str,
        target_logits: torch.Tensor,     # [1, K_valid, V]
    ):
        """Record verification counts at the vLLM verifier.

        Called BEFORE bolt_update, using target logits from rejection
        sampling. ``target_logits`` must contain only valid (unpadded)
        positions. Records accepted count, committed tokens, zero/full-block
        events.

        Note: acceptance is estimated by greedy argmax comparison against the
        teacher logits, which matches rejection sampling only for greedy
        decoding; with temperature/top-p sampling this is an approximation.
        """
        st = self.states.get(req_id)
        if (
            st is None
            or st.saved_draft is None
            or st.saved_round != self.propose_round
        ):
            return
        try:
            draft_tokens = st.saved_draft[0]  # [K]
            K_actual = min(draft_tokens.shape[0], target_logits.shape[1])
            if K_actual <= 0:
                return
            target_ids = target_logits[0, :K_actual].argmax(-1)  # [K_actual]
            # Count consecutive matches from position 0 (single host sync).
            # Cast to int32 before cumprod: NPU aclnnCumprod does not
            # support bool tensors (AclNN_Parameter_Error EZ1001).
            matches = (
                draft_tokens[:K_actual].to(target_ids.device) == target_ids
            ).to(torch.int32)
            accepted = int(matches.cumprod(0).sum().item())

            is_full_block = accepted >= K_actual

            st.verify_rounds += 1
            st.accepted_draft_tokens += accepted
            st.committed_tokens += accepted + 1
            if accepted == 0:
                st.zero_accept_rounds += 1
            if is_full_block:
                st.full_block_rounds += 1
            if accepted < len(st.accept_hist):
                st.accept_hist[accepted] += 1
            else:
                st.accept_hist[-1] += 1
            st.last_accepted = accepted

            if (
                self._debug_level >= 1
                and st.verify_rounds % self._debug_interval == 0
            ):
                rounds = st.verify_rounds
                logger.info(
                    "[BOLT][verify] req=%s round=%d accepted=%d/%d "
                    "tau=%.3f tau_bonus=%.3f zero_pct=%.1f full_pct=%.1f "
                    "updates=%d",
                    req_id, rounds, accepted, K_actual,
                    st.accepted_draft_tokens / rounds,
                    (st.accepted_draft_tokens + rounds) / rounds,
                    100.0 * st.zero_accept_rounds / rounds,
                    100.0 * st.full_block_rounds / rounds,
                    st.update_count,
                )
        except Exception:
            if not self._verify_error_logged:
                self._verify_error_logged = True
                logger.warning(
                    "BOLT: record_verify failed (suppressing further "
                    "warnings)", exc_info=True,
                )

    # ---- BOLT update ----

    def bolt_update_batch(
        self,
        req_ids: list[str],
        target_logits_list: list[torch.Tensor],
        valid_lens: list[int] | None = None,
    ):
        """Batched BOLT update for B >= 1 requests in ONE pass.

        Args:
            req_ids: list of request IDs with pending proposals.
            target_logits_list: list of [1, K, V] teacher logits per request
                (zero-padded to K rows when fewer draft tokens were verified).
            valid_lens: per-request count of valid (unpadded) rows in
                ``target_logits_list``. Padded rows are masked out of the
                gradient. When None, all K rows are treated as valid.
        """
        if not self.enabled:
            return
        if self.W2 is None:
            if not self._vocab_mismatch_logged:
                self._vocab_mismatch_logged = True
                logger.warning("BOLT: W2 not set, skipping update")
            return
        V_w2 = self.W2.shape[0]
        _t0 = time.perf_counter() if self._debug_level >= 1 else 0.0

        # Gather per-request data
        valid = []
        for i, rid in enumerate(req_ids):
            st = self.states.get(rid)
            if st is None or st.saved_e is None:
                continue
            if st.saved_round != self.propose_round:
                # Stale proposal (its drafts were never verified) — drop it.
                st.saved_e = None
                st.saved_base = None
                st.saved_draft = None
                continue
            tl = target_logits_list[i]
            if tl.dim() == 2:
                tl = tl.unsqueeze(0)
            elif tl.dim() == 3 and tl.shape[0] > 1:
                tl = tl[:1]
            # The analytical gradient projects both draft and teacher logits
            # through W2 [V, r]; it is only valid when both live in the same
            # vocab space as W2 (i.e. no draft->target vocab remapping).
            if tl.shape[-1] != V_w2 or st.saved_base.shape[-1] != V_w2:
                if not self._vocab_mismatch_logged:
                    self._vocab_mismatch_logged = True
                    logger.warning(
                        "BOLT: vocab size mismatch (W2=%d, base=%d, "
                        "target=%d); draft/target vocab remapping is not "
                        "supported, skipping update",
                        V_w2, st.saved_base.shape[-1], tl.shape[-1],
                    )
                continue
            vl = (
                valid_lens[i]
                if valid_lens is not None and i < len(valid_lens)
                else self.K
            )
            valid.append((rid, st, tl, vl))

        if not valid:
            return

        B = len(valid)
        # Clamp to the shortest proposal/teacher length actually available.
        K_actual = min(
            self.K,
            min(st.saved_e.shape[1] for _, st, _, _ in valid),
            min(tl.shape[1] for _, _, tl, _ in valid),
        )
        if K_actual <= 0:
            return

        # Stack all per-request tensors into batched [B, K, ...]
        e_batch = torch.cat(
            [st.saved_e[:, :K_actual, :] for _, st, _, _ in valid], 0
        )  # [B, K, r]
        base_batch = torch.cat(
            [st.saved_base[:, :K_actual, :] for _, st, _, _ in valid], 0
        )  # [B, K, V]
        tl_batch = torch.cat(
            [tl[:, :K_actual, :] for _, _, tl, _ in valid], 0
        )  # [B, K, V]

        # Alpha: [B, K] position weights, zeroed on padded (invalid) rows so
        # zero-padded teacher logits do not contribute to the gradient.
        alpha_batch = (
            self.alpha[:K_actual].unsqueeze(0).expand(B, -1).contiguous()
        )
        for b_i, (_, _, _, vl) in enumerate(valid):
            if vl < K_actual:
                alpha_batch[b_i, vl:] = 0.0

        # Multi-step update: recompute gradient on fixed proposal chain
        n_update_steps = self.config.update_steps
        for _step_i in range(n_update_steps):
            # Refresh deltas_batch with current deltas
            deltas_batch = torch.stack(
                [st.optimizer.delta for _, st, _, _ in valid], 0
            )  # [B, r, r]

            # Compute gradients (vanilla_residual is the only implemented
            # head mode; BoltConfig rejects 'bounded' at validation time).
            if self.config.loss == "cetv":
                grads = cetv_residual_gradient_batch(
                    e_batch, base_batch, deltas_batch, self.W2,
                    tl_batch, alpha_batch, self.anchor_lambda,
                    K_actual, self.config.ce_weight, self.config.tv_weight,
                )  # [B, r, r]
            else:  # kl
                grads = kl_residual_gradient_batch(
                    e_batch, base_batch, deltas_batch, self.W2,
                    tl_batch, alpha_batch, self.anchor_lambda, K_actual,
                )  # [B, r, r]

            # Apply per-req optimizer steps
            for i, (rid, st, _, _) in enumerate(valid):
                gnorm, clipped = st.optimizer.step(grads[i])
                st.update_count += 1
                st.grad_norm_sum += gnorm
                st.grad_norm_max = max(st.grad_norm_max, gnorm)
                st.clip_ratio_sum += clipped

        # Clear saved data
        for _, st, _, _ in valid:
            st.saved_e = None
            st.saved_base = None
            st.saved_draft = None

        if self._debug_level >= 1:
            # Wall time of the full batched update (gather + gradient +
            # optimizer steps). The per-step gnorm .item() calls force
            # host-device sync, so device compute is included.
            elapsed_ms = (time.perf_counter() - _t0) * 1000.0
            self._update_time_sum_ms += elapsed_ms
            self._update_time_count += 1
            self._update_time_max_ms = max(
                self._update_time_max_ms, elapsed_ms
            )
            if self._update_time_count % self._debug_interval == 0:
                logger.info(
                    "[BOLT][timing] update_batch #%d B=%d K=%d steps=%d "
                    "took %.2f ms (avg %.2f ms, max %.2f ms over %d calls)",
                    self._update_time_count, B, K_actual, n_update_steps,
                    elapsed_ms,
                    self._update_time_sum_ms / self._update_time_count,
                    self._update_time_max_ms, self._update_time_count,
                )

        if self._debug_level >= 2:
            for rid, st, _, _ in valid:
                dn = st.optimizer.delta.abs().max().item()
                logger.info(
                    "[BOLT][update] req=%s update#%d delta_max=%.6f "
                    "grad_norm_mean=%.4f grad_norm_max=%.4f clip_ratio=%.2f",
                    rid, st.update_count, dn,
                    st.grad_norm_sum / max(1, st.update_count),
                    st.grad_norm_max,
                    st.clip_ratio_sum / max(1, st.update_count),
                )

    # ---- Legacy / utility ----

    def get_metrics(self) -> dict:
        """Get metrics from the first active request (for diagnostics)."""
        for st in self.states.values():
            return self._compute_metrics(st)
        return self._compute_metrics(ReqState())

    def reset(self):
        """Clear all request states."""
        self.states.clear()