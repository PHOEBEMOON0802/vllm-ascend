#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for the V2 D-MTP2 speculator (worker/v2/spec_decode/dmtp2)."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as dflash2_spec

from vllm_ascend.worker.v2.spec_decode.dmtp2.speculator import AscendDMTP2Speculator


def _build_speculator(*, num_speculative_steps: int, max_num_reqs: int = 4, top_k: int = 3):
    """Construct without the real draft model / NPU wiring."""

    def fake_base_init(self, vllm_config, device):
        self.num_speculative_steps = num_speculative_steps
        self.num_query_per_req = 1 + num_speculative_steps
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = 128
        self.hidden_size = 8
        self.dtype = torch.float32
        self.device = device
        self.top_k = top_k
        self.enable_adaptive_verification = False
        self.draft_logits = None
        self.parallel_drafting_token_id = 0
        self.candidate_sampler = dflash2_spec.CandidateSampler(
            max_num_reqs, num_speculative_steps, top_k, device
        )

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1, use_ubatching=False, enable_eplb=False
        )
    )
    with patch.object(dflash2_spec.DFlash2Speculator, "__init__", fake_base_init):
        return AscendDMTP2Speculator(vllm_config, torch.device("cpu"))


@pytest.mark.parametrize("num_speculative_steps", [2, 3, 7])
def test_candidate_sampler_matches_drafted_steps(num_speculative_steps):
    speculator = _build_speculator(num_speculative_steps=num_speculative_steps)
    assert speculator.num_sample_steps == num_speculative_steps - 1
    # The walk kernel strides by CandidateSampler.num_steps over the
    # re-indexed buffers, so it must match num_sample_steps, not the
    # base-class num_speculative_steps.
    assert speculator.candidate_sampler.num_steps == speculator.num_sample_steps
    assert speculator.candidate_sampler.top_k == speculator.top_k


@pytest.mark.parametrize("num_speculative_steps", [2, 4])
def test_sample_buffers_select_slots_after_t1(num_speculative_steps):
    speculator = _build_speculator(num_speculative_steps=num_speculative_steps)
    num_reqs = speculator.max_num_reqs
    width = num_reqs * num_speculative_steps
    # prepare_dflash_inputs numbers the sampled slots 1..N request-major.
    speculator.sample_indices = torch.arange(width, dtype=torch.int64)
    speculator.sample_pos = torch.arange(width, dtype=torch.int64) + 100
    speculator.sample_idx_mapping = torch.arange(num_reqs, dtype=torch.int64).repeat_interleave(
        num_speculative_steps
    )

    indices, positions, mapping = speculator._dmtp_sample_buffers(num_reqs)

    keep = torch.tensor([k for k in range(1, num_speculative_steps)])
    expected = (torch.arange(num_reqs)[:, None] * num_speculative_steps + keep[None, :]).reshape(-1)
    assert torch.equal(indices, expected)
    assert torch.equal(positions, expected + 100)
    # idx_mapping is per-request broadcast: entry i belongs to request i // (N-1).
    assert torch.equal(
        mapping, torch.arange(num_reqs).repeat_interleave(num_speculative_steps - 1)
    )
    assert indices.numel() == num_reqs * (num_speculative_steps - 1)


def test_propose_t1_uses_pinned_prepare_prefill_inputs():
    """_propose_t1 must not hand-roll the shift on whole-batch tensors.

    The earlier plain-tensor port broadcast ``next_token[req]`` (num_reqs)
    against per-row tensors (num_tokens) and crashed once the two differed.
    The pinned kernel does the shift per request; assert we call it.
    """
    speculator = _build_speculator(num_speculative_steps=3)
    num_reqs = 2
    num_tokens = 6

    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.tensor([11, 99, 0, 21, 22, 66], dtype=torch.int64),
        positions=torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64),
    )
    speculator.block_tables = SimpleNamespace(
        slot_mappings=torch.zeros(1, num_tokens, dtype=torch.int64)
    )
    speculator.kv_cache_config = object()
    speculator._group_causal = False
    speculator._build_attn_metadata = MagicMock(return_value={})
    speculator.model = MagicMock()
    speculator.model.mtp_forward.return_value = torch.zeros(
        num_tokens, speculator.hidden_size
    )
    speculator.model.compute_t1_logits.return_value = torch.zeros(num_reqs, 5)

    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        query_start_loc_np=[0, 3, 6],
        seq_lens_cpu_upper_bound=torch.tensor([3, 3], dtype=torch.int32),
        idx_mapping=torch.tensor([0, 1], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int64),
    )
    num_sampled = torch.tensor([1, 0], dtype=torch.int64)
    num_rejected = torch.tensor([1, 0], dtype=torch.int64)
    last_sampled = torch.tensor([99, 88], dtype=torch.int64)
    next_prefill = torch.tensor([77, 66], dtype=torch.int64)

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.dmtp2.speculator.prepare_prefill_inputs"
        ) as prepare,
        patch(
            "vllm_ascend.worker.v2.spec_decode.dmtp2.speculator.build_slot_mappings_by_layer",
            return_value={},
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.dmtp2.speculator.set_forward_context",
            side_effect=lambda *a, **k: nullcontext(),
        ),
    ):
        speculator._propose_t1(
            input_batch,
            {},
            {},
            torch.zeros(num_tokens, speculator.hidden_size),
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill,
        )

    prepare.assert_called_once()
    # The MTP forward reads the buffers the kernel wrote, not a batched shift.
    mtp_ids = speculator.model.mtp_forward.call_args.kwargs["input_ids"]
    assert torch.equal(mtp_ids, speculator.input_buffers.input_ids[:num_tokens])


def test_init_cudagraph_manager_forwards_requested_mode(monkeypatch):
    """Only the DFlash2 pass is captured, so the requested graph mode reaches
    the shared DFlash ACL-graph manager unchanged."""
    calls: list[CUDAGraphMode] = []
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dflash.speculator.AscendDFlashSpeculator.init_cudagraph_manager",
        lambda self, mode: calls.append(mode),
    )
    speculator = AscendDMTP2Speculator.__new__(AscendDMTP2Speculator)

    speculator.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert calls == [CUDAGraphMode.FULL_DECODE_ONLY]


def test_init_rejects_single_speculative_token():
    def fake_base_init(self, vllm_config, device):
        self.num_speculative_steps = 1
        self.num_query_per_req = 2
        self.max_num_reqs = 2
        self.max_num_tokens = 8
        self.hidden_size = 4
        self.dtype = torch.float32
        self.device = device
        self.top_k = 2
        self.enable_adaptive_verification = False
        self.draft_logits = None
        self.parallel_drafting_token_id = 0
        self.candidate_sampler = MagicMock()

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1, use_ubatching=False, enable_eplb=False
        )
    )
    with (
        patch.object(dflash2_spec.DFlash2Speculator, "__init__", fake_base_init),
        pytest.raises(ValueError, match="num_speculative_tokens >= 2"),
    ):
        AscendDMTP2Speculator(vllm_config, torch.device("cpu"))
