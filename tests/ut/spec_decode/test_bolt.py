# SPDX-License-Identifier: Apache-2.0
"""Unit tests for BOLT (bolt_gradient / bolt_manager), CPU-only."""
import json
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.spec_decode import bolt_manager as bolt_manager_module
from vllm_ascend.spec_decode.bolt_gradient import (
    cetv_residual_gradient_batch,
    exact_projection,
    kl_residual_gradient_batch,
    position_weights,
)
from vllm_ascend.spec_decode.bolt_manager import (
    BoltManager,
    RequestAdam,
    RequestMuon,
    RequestSGD,
)

R, D, K, V = 8, 16, 4, 32


def make_config(**overrides):
    cfg = dict(
        enabled=True,
        optimizer="sgdm",
        lr=0.01,
        gamma=4.0,
        anchor_lambda=0.5,
        loss="cetv",
        ce_weight=0.5,
        tv_weight=0.5,
        head_mode="vanilla_residual",
        sgd_momentum=0.9,
        clip=1.0,
        metrics_dir="",
        update_steps=1,
        update_interval=1,
        debug=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def make_manager(**overrides) -> BoltManager:
    mgr = BoltManager(make_config(**overrides), r=R, d=D, K=K, device="cpu")
    W1 = torch.randn(V, R)
    W2 = torch.randn(V, R)
    mgr.set_weights(W1=W1, W2=W2)
    return mgr


class TestGradient:

    def test_position_weights(self):
        w = position_weights(K, 4.0, device="cpu", dtype=torch.float32)
        assert w.shape == (K,)
        assert torch.allclose(w, torch.exp(-torch.arange(K) / 4.0))

    def test_exact_projection_matches_direct(self):
        torch.manual_seed(0)
        logits = torch.randn(2, K, V)
        W2 = torch.randn(V, R)
        # chunk smaller than V to exercise the chunked path
        chunked = exact_projection(logits, W2, chunk_size=5)
        direct = torch.softmax(logits.float(), -1) @ W2.float()
        assert torch.allclose(chunked, direct, atol=1e-5)

    @pytest.mark.parametrize("fn", ["cetv", "kl"])
    def test_gradient_shape_and_finite(self, fn):
        torch.manual_seed(0)
        B = 3
        e = torch.randn(B, K, R)
        base = torch.randn(B, K, V)
        deltas = torch.zeros(B, R, R)
        W2 = torch.randn(V, R)
        tl = torch.randn(B, K, V)
        alpha = position_weights(K, 4.0, device="cpu",
                                 dtype=torch.float32).expand(B, -1)
        if fn == "cetv":
            g = cetv_residual_gradient_batch(e, base, deltas, W2, tl, alpha,
                                             0.5, K, 0.5, 0.5)
        else:
            g = kl_residual_gradient_batch(e, base, deltas, W2, tl, alpha,
                                           0.5, K)
        assert g.shape == (B, R, R)
        assert torch.isfinite(g).all()

    def test_alpha_mask_excludes_padded_rows(self):
        """Zeroed alpha rows must not contribute to the gradient."""
        torch.manual_seed(0)
        e = torch.randn(1, K, R)
        base = torch.randn(1, K, V)
        deltas = torch.zeros(1, R, R)
        W2 = torch.randn(V, R)
        tl = torch.randn(1, K, V)
        alpha = position_weights(K, 4.0, device="cpu",
                                 dtype=torch.float32).unsqueeze(0).clone()
        valid = 2
        alpha[:, valid:] = 0.0
        g_masked = cetv_residual_gradient_batch(e, base, deltas, W2, tl,
                                                alpha, 0.5, K, 0.5, 0.5)
        # Corrupt the padded rows: the gradient must be unchanged.
        tl2 = tl.clone()
        tl2[:, valid:] = 123.0
        g_masked2 = cetv_residual_gradient_batch(e, base, deltas, W2, tl2,
                                                 alpha, 0.5, K, 0.5, 0.5)
        assert torch.allclose(g_masked, g_masked2, atol=1e-5)

    def test_kl_gradient_matches_autograd(self):
        """The analytical KL gradient is exactly the autograd gradient of
        L = sum_k alpha_k * [KL(p_T || q) + anchor * KL(p_ref || q)]
        (up to entropy constants). Locks the e.T @ du orientation."""
        torch.manual_seed(5)
        anchor = 0.3
        e = torch.randn(1, K, R)
        base = torch.randn(1, K, V)
        W2 = torch.randn(V, R)
        tl = torch.randn(1, K, V)
        alpha = position_weights(K, 4.0, device="cpu",
                                 dtype=torch.float32).unsqueeze(0)
        delta0 = torch.randn(R, R) * 0.1

        g = kl_residual_gradient_batch(e, base, delta0.unsqueeze(0), W2, tl,
                                       alpha, anchor, K)[0]

        delta = delta0.clone().requires_grad_(True)
        q_logits = base + (e + e @ delta) @ W2.t()
        log_q = torch.log_softmax(q_logits, -1)
        p_t = torch.softmax(tl, -1)
        p_ref = torch.softmax(base + e @ W2.t(), -1)
        loss = (alpha * (-(p_t * log_q).sum(-1)
                         - anchor * (p_ref * log_q).sum(-1))).sum()
        loss.backward()
        assert torch.allclose(g, delta.grad, atol=1e-4)


class TestOptimizers:

    @pytest.mark.parametrize("cls", [RequestMuon, RequestSGD, RequestAdam])
    def test_step_updates_delta_and_version(self, cls):
        opt = cls((R, R), "cpu", lr=0.01)
        g = torch.randn(R, R)
        gnorm, clipped = opt.step(g)
        assert opt.version == 1
        assert opt.step_count == 1
        assert gnorm > 0
        assert not torch.equal(opt.delta, torch.zeros(R, R))

    def test_clipping_reported(self):
        opt = RequestSGD((R, R), "cpu", lr=0.01, clip=1e-6)
        _, clipped = opt.step(torch.ones(R, R))
        assert clipped == 1.0


class TestManagerLifecycle:

    def test_new_and_finished(self):
        mgr = make_manager()
        mgr.on_request_new("a")
        assert "a" in mgr.states
        mgr.on_request_finished("a")
        assert "a" not in mgr.states
        # Finishing an unknown request is a no-op.
        mgr.on_request_finished("ghost")

    def test_save_proposal_requires_state(self):
        """save_proposal must not implicitly create request state."""
        mgr = make_manager()
        mgr.save_proposal("nope", torch.zeros(1, K, R), torch.zeros(1, K, V),
                          torch.zeros(1, K, dtype=torch.long))
        assert "nope" not in mgr.states

    def test_get_delta_disabled_returns_none(self):
        mgr = make_manager(enabled=False)
        mgr.on_request_new("a")
        assert mgr.get_delta("a") is None

    def test_metrics_enabled_property(self, tmp_path):
        assert not make_manager().metrics_enabled
        assert make_manager(metrics_dir=str(tmp_path)).metrics_enabled

    def test_metrics_file_written(self, tmp_path):
        mgr = make_manager(metrics_dir=str(tmp_path))
        mgr.on_request_new("a")
        mgr.on_request_finished("a")
        path = os.path.join(str(tmp_path), "metrics_a.json")
        assert os.path.exists(path)
        with open(path) as f:
            m = json.load(f)
        assert m["req_id"] == "a"
        assert m["verify_rounds"] == 0


class TestRecordVerify:

    def test_consecutive_accept_count(self):
        mgr = make_manager()
        mgr.on_request_new("a")
        draft = torch.tensor([[3, 5, 7, 9]])
        mgr.save_proposal("a", torch.zeros(1, K, R), torch.zeros(1, K, V),
                          draft)
        # Teacher argmax agrees on positions 0,1, disagrees on 2.
        tl = torch.zeros(1, K, V)
        tl[0, 0, 3] = 10.0
        tl[0, 1, 5] = 10.0
        tl[0, 2, 8] = 10.0  # mismatch (draft has 7)
        tl[0, 3, 9] = 10.0  # would match but comes after the break
        mgr.record_verify("a", tl)
        st = mgr.states["a"]
        assert st.verify_rounds == 1
        assert st.accepted_draft_tokens == 2
        assert st.committed_tokens == 3  # accepted + bonus token
        assert st.accept_hist[2] == 1

    def test_full_block(self):
        mgr = make_manager()
        mgr.on_request_new("a")
        draft = torch.tensor([[1, 2, 3, 4]])
        mgr.save_proposal("a", torch.zeros(1, K, R), torch.zeros(1, K, V),
                          draft)
        tl = torch.zeros(1, K, V)
        for k, tok in enumerate([1, 2, 3, 4]):
            tl[0, k, tok] = 10.0
        mgr.record_verify("a", tl)
        assert mgr.states["a"].full_block_rounds == 1

    def test_without_saved_draft_is_noop(self):
        mgr = make_manager()
        mgr.on_request_new("a")
        mgr.record_verify("a", torch.zeros(1, K, V))
        assert mgr.states["a"].verify_rounds == 0

    def test_stale_round_is_noop(self):
        mgr = make_manager()
        mgr.on_request_new("a")
        mgr.save_proposal("a", None, None, torch.tensor([[1, 2, 3, 4]]))
        mgr.states["a"].saved_round = -5  # simulate a stale round tag
        mgr.record_verify("a", torch.zeros(1, K, V))
        assert mgr.states["a"].verify_rounds == 0

    def test_saved_draft_is_cloned(self):
        """The caller's draft buffer is reused in-place across propose
        cycles; saved tokens must not alias it."""
        mgr = make_manager()
        mgr.on_request_new("a")
        buf = torch.tensor([[1, 2, 3, 4]])
        mgr.save_proposal("a", None, None, buf)
        buf[0, 0] = 99
        assert mgr.states["a"].saved_draft[0, 0].item() == 1

    def test_metrics_only_mode(self):
        """enabled=False + metrics_dir set: verify metrics still work with
        e/base omitted from save_proposal."""
        mgr = make_manager(enabled=False, metrics_dir="unused_dir")
        assert mgr.metrics_enabled
        mgr.on_request_new("a")
        mgr.save_proposal("a", None, None, torch.tensor([[1, 2, 3, 4]]))
        tl = torch.zeros(1, K, V)
        tl[0, 0, 1] = 10.0  # only the first draft token matches
        mgr.record_verify("a", tl)
        st = mgr.states["a"]
        assert st.verify_rounds == 1
        assert st.accepted_draft_tokens == 1
        assert st.saved_e is None  # no update tensors retained


class TestUpdateBatch:

    def _propose(self, mgr, rid, seed=0):
        torch.manual_seed(seed)
        e = torch.randn(1, K, R)
        base = torch.randn(1, K, V)
        draft = torch.randint(0, V, (1, K))
        mgr.on_request_new(rid)
        mgr.save_proposal(rid, e, base, draft)

    @pytest.mark.parametrize("optimizer", ["muon", "sgdm", "adam"])
    @pytest.mark.parametrize("loss", ["cetv", "kl"])
    def test_update_changes_delta(self, optimizer, loss):
        mgr = make_manager(optimizer=optimizer, loss=loss)
        self._propose(mgr, "a")
        tl = torch.randn(1, K, V)
        mgr.bolt_update_batch(["a"], [tl], [K])
        st = mgr.states["a"]
        assert st.update_count == 1
        assert st.optimizer.delta.abs().max().item() > 0
        assert torch.isfinite(st.optimizer.delta).all()
        # Saved proposal is consumed.
        assert st.saved_e is None and st.saved_base is None

    def test_update_batch_multiple_requests(self):
        mgr = make_manager()
        self._propose(mgr, "a", seed=1)
        self._propose(mgr, "b", seed=2)
        mgr.bolt_update_batch(["a", "b"],
                              [torch.randn(1, K, V), torch.randn(1, K, V)],
                              [K, K])
        assert mgr.states["a"].update_count == 1
        assert mgr.states["b"].update_count == 1
        assert not torch.equal(mgr.states["a"].optimizer.delta,
                               mgr.states["b"].optimizer.delta)

    def test_padded_rows_do_not_affect_update(self):
        """Two managers, identical valid rows, different garbage in padded
        rows -> identical deltas."""
        results = []
        for garbage in (0.0, 99.0):
            mgr = make_manager()
            torch.manual_seed(7)
            mgr.set_weights(W1=torch.randn(V, R), W2=torch.randn(V, R))
            e = torch.randn(1, K, R)
            base = torch.randn(1, K, V)
            draft = torch.randint(0, V, (1, K))
            mgr.on_request_new("a")
            mgr.save_proposal("a", e, base, draft)
            tl = torch.randn(1, K, V)
            tl[:, 2:] = garbage  # only first 2 rows valid
            mgr.bolt_update_batch(["a"], [tl], [2])
            results.append(mgr.states["a"].optimizer.delta.clone())
        assert torch.allclose(results[0], results[1], atol=1e-6)

    def test_vocab_mismatch_skipped(self):
        mgr = make_manager()
        self._propose(mgr, "a")
        tl = torch.randn(1, K, V + 8)  # wrong vocab size
        mgr.bolt_update_batch(["a"], [tl], [K])
        assert mgr.states["a"].update_count == 0

    def test_disabled_is_noop(self):
        mgr = make_manager(enabled=False)
        mgr.on_request_new("a")
        st = mgr.states["a"]
        st.saved_e = torch.randn(1, K, R)
        st.saved_base = torch.randn(1, K, V)
        mgr.bolt_update_batch(["a"], [torch.randn(1, K, V)], [K])
        assert st.update_count == 0

    def test_update_steps_multi(self):
        mgr = make_manager(update_steps=3)
        self._propose(mgr, "a")
        mgr.bolt_update_batch(["a"], [torch.randn(1, K, V)], [K])
        assert mgr.states["a"].update_count == 3

    def test_begin_round_invalidates_stale_proposals(self):
        """A proposal that was never verified must not be paired with a
        later round's target logits."""
        mgr = make_manager()
        self._propose(mgr, "a")
        mgr.begin_round()  # next propose cycle starts; "a" skipped saving
        st = mgr.states["a"]
        assert st.saved_e is None  # stale tensors released
        mgr.bolt_update_batch(["a"], [torch.randn(1, K, V)], [K])
        assert st.update_count == 0

    def _one_cycle(self, mgr, rid):
        """save_proposal + bolt_update_batch + begin_round (prod order)."""
        torch.manual_seed(0)
        mgr.save_proposal(rid, torch.randn(1, K, R), torch.randn(1, K, V),
                          torch.randint(0, V, (1, K)))
        mgr.bolt_update_batch([rid], [torch.randn(1, K, V)], [K])
        mgr.begin_round()

    def test_update_interval_throttles(self):
        """interval=4 over 8 blocks -> updates on blocks 1 and 5 only;
        skipped blocks release their saved tensors."""
        mgr = make_manager(update_interval=4)
        mgr.on_request_new("a")
        st = mgr.states["a"]
        expected = [1, 1, 1, 1, 2, 2, 2, 2]
        for i in range(8):
            self._one_cycle(mgr, "a")
            assert st.update_count == expected[i]
            assert st.saved_e is None  # released whether updated or skipped
        assert st.update_opportunities == 8

    def test_update_interval_one_is_default_behavior(self):
        mgr = make_manager(update_interval=1)
        mgr.on_request_new("a")
        for i in range(3):
            self._one_cycle(mgr, "a")
        assert mgr.states["a"].update_count == 3

    def test_delta_applied_direction_reduces_loss(self):
        """One SGD step on a fixed objective should reduce the CE+anchor
        loss used by the analytical gradient (sanity, small lr)."""
        mgr = make_manager(optimizer="sgdm", sgd_momentum=0.0, lr=0.05,
                           anchor_lambda=0.0, ce_weight=1.0, tv_weight=0.0)
        torch.manual_seed(3)
        W2 = torch.randn(V, R)
        mgr.set_weights(W1=torch.randn(V, R), W2=W2)
        e = torch.randn(1, K, R)
        base = torch.randn(1, K, V)
        tl = torch.randn(1, K, V) * 3
        alpha = mgr.alpha[:K].unsqueeze(0)

        def ce_loss(delta):
            bias = (e + e @ delta) @ W2.t()
            q_proj = exact_projection(base + bias, W2)
            t_proj = exact_projection(tl, W2)
            per_pos = 0.5 * ((q_proj - t_proj) ** 2).sum(-1)
            return (per_pos * alpha).sum().item()

        mgr.on_request_new("a")
        mgr.save_proposal("a", e, base, torch.randint(0, V, (1, K)))
        before = ce_loss(torch.zeros(R, R))
        mgr.bolt_update_batch(["a"], [tl], [K])
        after = ce_loss(mgr.states["a"].optimizer.delta)
        assert after < before


class TestDebugEnv:

    def test_debug_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_BOLT_DEBUG", raising=False)
        assert make_manager()._debug_level == 0

    def test_debug_level_and_interval_from_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_BOLT_DEBUG", "1")
        monkeypatch.setenv("VLLM_ASCEND_BOLT_DEBUG_INTERVAL", "5")
        mgr = make_manager()
        assert mgr._debug_level == 1
        assert mgr._debug_interval == 5

    def test_config_debug_implies_level2(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_BOLT_DEBUG", raising=False)
        assert make_manager(debug=True)._debug_level == 2

    def test_garbage_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_BOLT_DEBUG", "yes")
        monkeypatch.setenv("VLLM_ASCEND_BOLT_DEBUG_INTERVAL", "oops")
        mgr = make_manager()
        assert mgr._debug_level == 0
        assert mgr._debug_interval == 1

    def test_debug_smoke_full_cycle(self, monkeypatch):
        """Debug printing must not break verify/update/finish, and must
        emit the expected lines at level 2."""
        monkeypatch.setenv("VLLM_ASCEND_BOLT_DEBUG", "2")
        monkeypatch.setenv("VLLM_ASCEND_BOLT_DEBUG_INTERVAL", "1")
        mgr = make_manager()
        mgr.on_request_new("a")
        torch.manual_seed(0)
        mgr.save_proposal("a", torch.randn(1, K, R), torch.randn(1, K, V),
                          torch.randint(0, V, (1, K)))
        with mock.patch.object(bolt_manager_module.logger, "info") as m:
            mgr.record_verify("a", torch.randn(1, K, V))
            mgr.bolt_update_batch(["a"], [torch.randn(1, K, V)], [K])
            mgr.on_request_finished("a")
        logged = " | ".join(str(c) for c in m.call_args_list)
        assert "[BOLT][verify]" in logged
        assert "[BOLT][update]" in logged
        assert "[BOLT][finish]" in logged
        assert "[BOLT][timing]" in logged

    def test_timing_respects_interval(self, monkeypatch):
        """Timing line printed every N update calls; stats still accumulate."""
        monkeypatch.setenv("VLLM_ASCEND_BOLT_DEBUG", "1")
        monkeypatch.setenv("VLLM_ASCEND_BOLT_DEBUG_INTERVAL", "2")
        mgr = make_manager()
        with mock.patch.object(bolt_manager_module.logger, "info") as m:
            for i in range(4):
                mgr.on_request_new("a")
                mgr.save_proposal("a", torch.randn(1, K, R),
                                  torch.randn(1, K, V),
                                  torch.randint(0, V, (1, K)))
                mgr.bolt_update_batch(["a"], [torch.randn(1, K, V)], [K])
        timing_lines = [
            c for c in m.call_args_list if "[BOLT][timing]" in str(c)
        ]
        assert len(timing_lines) == 2  # calls 2 and 4 of 4
        assert mgr._update_time_count == 4
        assert mgr._update_time_sum_ms > 0

    def test_no_debug_lines_when_disabled(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_BOLT_DEBUG", raising=False)
        mgr = make_manager()
        mgr.on_request_new("a")
        mgr.save_proposal("a", torch.randn(1, K, R), torch.randn(1, K, V),
                          torch.randint(0, V, (1, K)))
        with mock.patch.object(bolt_manager_module.logger, "info") as m:
            mgr.record_verify("a", torch.randn(1, K, V))
            mgr.bolt_update_batch(["a"], [torch.randn(1, K, V)], [K])
            mgr.on_request_finished("a")
        logged = " | ".join(str(c) for c in m.call_args_list)
        assert "[BOLT][verify]" not in logged
        assert "[BOLT][update]" not in logged
        assert "[BOLT][finish]" not in logged
