#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""D-MTP2 speculator on the Ascend V2 (MRV2) model runner.

Port of the upstream reference speculator. Each ``propose()`` runs two draft
passes:

1. **MTP pass** (eagle-style draft prefill over the target batch): the shifted
   target tokens paired with the target's last hidden states go through the
   frozen MTP block, keeping its KV cache in sync; each request's last valid
   row yields ``h_MTP``, whose greedy argmax is T1.
2. **DFlash2 pass**: the usual parallel block pass with the dual-anchor layout
   ``[A, T1, MASK x (B-2)]`` -- slot 1 carries T1 with the fusion residual
   added to its embedding, slots 2+ are drafted, and the candidate-selector
   walk is seeded with T1 as the first predecessor.

Bring-up limits (guarded with clear errors at construction):

- drafting runs eagerly (the MTP pass and the T1 injection are not captured);
- greedy draft sampling only;
- no data parallelism / EPLB for the MTP MoE forward.

The pinned vLLM's ``prepare_dflash_inputs`` has no ``first_draft_offset``
switch, so this speculator re-indexes the sample buffers it produced
(``sample_indices`` / ``sample_pos`` / ``sample_idx_mapping``) from "slots
1..N" to "slots 2..N" instead of changing the kernel.
"""

from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as _dflash2_spec

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
    AscendDFlash2Speculator,
)


class AscendDMTP2Speculator(AscendDFlash2Speculator):
    """D-MTP2 drafter: frozen-MTP first token + DFlash2 continuation."""

    _speculator_name = "DMTP2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        if self.num_speculative_steps < 2:
            raise ValueError(
                "D-MTP2 needs num_speculative_tokens >= 2 (T1 + at least one "
                f"DFlash2 slot); got {self.num_speculative_steps}."
            )
        if self.enable_adaptive_verification:
            raise NotImplementedError(
                "D-MTP2 does not support adaptive verification yet."
            )
        if getattr(self, "draft_logits", None) is not None:
            raise NotImplementedError(
                "D-MTP2 only supports draft_sample_method='greedy' for now."
            )
        if vllm_config.parallel_config.data_parallel_size > 1:
            raise NotImplementedError(
                "D-MTP2's MTP pass does not synchronize token counts across "
                "data-parallel ranks yet."
            )

        # Dual-anchor layout: slot 0 is the bonus anchor, slot 1 is T1, and only
        # the remaining block slots are drafted by the DFlash2 pass.
        self.num_sample_steps = self.num_speculative_steps - 1

        self._fusion_rows = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
            + 1
        )
        self.t1_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.t1_hidden = torch.zeros(
            self.max_num_reqs, self.hidden_size, dtype=self.dtype, device=device
        )
        self.walk_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_sample_steps,
            dtype=torch.int64,
            device=device,
        )
        self.mtp_last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        # Private buffers for the shifted MTP inputs, so the draft input
        # buffers stay untouched for ``prepare_dflash_inputs``.
        self._mtp_input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self._mtp_positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

    # -- bring-up ---------------------------------------------------------

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # The MTP pass and the T1 injection between the two draft passes are
        # not captured yet, so the draft runs eagerly.
        super().init_cudagraph_manager(CUDAGraphMode.NONE)

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        model = super().load_draft_model(target_model, target_attn_layer_names)
        # The MTP block embeds the shifted target tokens with the verifier's
        # embedding, which the draft already shares with the target.
        model.mtp.embed_tokens = model.model.embed_tokens
        return model

    # -- MTP first-token pass ---------------------------------------------

    @torch.inference_mode()
    def _mtp_inputs(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-request one-token shift of the target batch for the MTP pass.

        The MTP predictor consumes ``(h_t, emb(x_{t+1}))`` per row, so each
        request's scheduled tokens are shifted by one inside its own query
        segment and the last row carries the next token to be sampled/prefilled.

        This mirrors the pinned vLLM's ``prepare_prefill_inputs`` kernel but is
        written as plain tensor ops: the pinned tree this port targets does not
        ship the ``spec_decode.autoregressive`` subpackage that the reference
        implementation imports from.
        """
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens

        query_start = input_batch.query_start_loc[:num_reqs]
        query_end = input_batch.query_start_loc[1 : num_reqs + 1]
        # Rejected draft tokens are not part of the kept prefix.
        row_end = query_end - num_rejected[:num_reqs]

        state = input_batch.idx_mapping[:num_reqs]
        next_token = torch.where(
            num_sampled[:num_reqs] > 0,
            last_sampled[state],
            next_prefill_tokens[state],
        ).to(torch.int64)

        rows = torch.arange(num_tokens, device=self.device)
        # Request owning each row; rows past a request's kept prefix are dropped.
        req = torch.searchsorted(query_start, rows, right=True) - 1
        req = req.clamp(min=0)
        owned = (req < num_reqs) & (rows < row_end[req])

        shifted = input_batch.input_ids[(rows + 1).clamp(max=num_tokens - 1)].to(
            torch.int64
        )
        is_last_row = owned & (rows + 1 >= row_end[req])
        ids = torch.where(is_last_row, next_token[req], shifted)
        ids = torch.where(owned, ids, torch.zeros_like(ids))

        positions = input_batch.positions[:num_tokens].to(torch.int64)
        self.mtp_last_token_indices[:num_reqs] = row_end - 1
        return ids, positions, row_end

    @torch.inference_mode()
    def _propose_t1(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],  # noqa: ARG002
        slot_mappings: dict[str, torch.Tensor],  # noqa: ARG002
        last_hidden_states: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
    ) -> None:
        """Run the frozen MTP block over the target batch.

        Keeps the MTP KV cache in sync and proposes T1 (+ ``h_MTP``) for every
        request. Runs before ``prepare_dflash_inputs``, which overwrites the
        draft input buffers.
        """
        num_reqs = input_batch.num_reqs
        num_tokens_padded = input_batch.num_tokens_after_padding

        mtp_ids, mtp_positions, _ = self._mtp_inputs(
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
        )
        self._mtp_input_ids[: mtp_ids.shape[0]] = mtp_ids
        self._mtp_positions[: mtp_positions.shape[0]] = mtp_positions

        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens_padded,
            num_reqs=num_reqs,
        )
        # The MTP block sees the target batch, so its metadata follows the
        # target's variable-length query layout rather than the draft's
        # uniform 1+N one.
        with build_attn_metadata_wrapper():
            mtp_attn_metadata = self._build_attn_metadata(
                num_reqs=num_reqs,
                batch_desc=batch_desc,
                query_start_loc_np=input_batch.query_start_loc_np,
                seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
                step=1,
                causal=self._group_causal,
            )
        mtp_slot_mappings = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_tokens_padded],
            self.kv_cache_config,
        )
        with set_forward_context(
            mtp_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens_padded,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=mtp_slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens_padded),
        ):
            mtp_hidden = self.model.mtp_forward(
                input_ids=self._mtp_input_ids[:num_tokens_padded],
                positions=self._mtp_positions[:num_tokens_padded],
                hidden_states=last_hidden_states[:num_tokens_padded],
            )

        rows = self.mtp_last_token_indices[:num_reqs]
        h_mtp = mtp_hidden[rows]
        self.t1_hidden[:num_reqs] = h_mtp
        t1_logits = self.model.compute_t1_logits(h_mtp)
        self.t1_tokens[:num_reqs] = t1_logits.argmax(dim=-1)

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        if dummy_run:
            # Profiling paths skip the real MTP pass; the fusion then adds a
            # zero-normed residual to a mask-token slot, which is shape-exact.
            self.t1_tokens[:num_reqs] = self.parallel_drafting_token_id
            self.t1_hidden[:num_reqs].zero_()
        else:
            # Runs even when no drafts are needed this step: the MTP KV cache
            # must stay in sync with the accepted tokens.
            self._propose_t1(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
            )
        return super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync_state=dp_sync_state,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
            num_speculative_tokens=num_speculative_tokens,
        )

    # -- DFlash2 pass with T1 injected -------------------------------------

    def _dmtp_sample_buffers(
        self, num_reqs: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-index the kernel-produced sample buffers from slots 1..N to 2..N.

        ``prepare_dflash_inputs`` numbers the sampled slots ``1 + k`` for
        ``k in [0, N)``. DMTP2 drafts only ``N - 1`` of them (slot 1 is the
        externally proposed T1), so the first entry of every request is
        dropped -- the pinned kernel has no ``first_draft_offset`` switch.
        """
        base = torch.arange(num_reqs, device=self.device) * self.num_speculative_steps
        src = (
            base[:, None] + torch.arange(1, self.num_speculative_steps, device=self.device)[None, :]
        ).reshape(-1)
        return (
            self.sample_indices[src],
            self.sample_pos[src],
            self.sample_idx_mapping[src],
        )

    @torch.inference_mode()
    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        num_query_tokens = num_reqs * self.num_query_per_req
        input_ids = self.input_buffers.input_ids
        t1 = self.t1_tokens[:num_reqs]

        # Inject T1 into slot 1 of every block (prepare_dflash_inputs wrote the
        # mask token there) and add the fusion residual to its embedding.
        input_ids[1:num_query_tokens:self.num_query_per_req] = t1
        inputs_embeds = self.model.embed_input_ids(input_ids[:num_tokens_padded])
        fusion_rows = self._fusion_rows[:num_reqs]
        inputs_embeds[fusion_rows] = self.model.fuse_t1_embedding(
            inputs_embeds[fusion_rows], self.t1_hidden[:num_reqs]
        ).to(inputs_embeds.dtype)

        last_hidden_states = self._run_model_with_embeds(
            num_tokens_padded,
            inputs_embeds,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )

        sample_indices, sample_pos, sample_idx_mapping = self._dmtp_sample_buffers(
            num_reqs
        )
        hidden_states = last_hidden_states[sample_indices].view(
            num_reqs, self.num_sample_steps, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(num_reqs, self.num_sample_steps, self.selector_top_k)
        unary_logits = unary_logits.view_as(candidate_ids)
        # The first walked slot's predecessor is T1, not the anchor.
        scores = self.model.model.candidate_selector(
            candidate_ids, unary_logits, hidden_states, t1
        )
        self._sample_path_with(
            candidate_ids, scores, num_reqs, sample_pos, sample_idx_mapping
        )
        self.draft_tokens[:num_reqs, 0] = t1
        self.draft_tokens[:num_reqs, 1:] = self.walk_tokens[:num_reqs]

    def _sample_path_with(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
        sample_pos: torch.Tensor,
        sample_idx_mapping: torch.Tensor,
    ) -> None:
        """``DFlash2Speculator._sample_path`` with the DMTP2 step count.

        The Ascend selector-walk kernel is patched over upstream's module
        global, so it is looked up at call time.
        """
        from vllm.triton_utils import triton

        block_k = triton.next_power_of_2(self.selector_top_k)
        _dflash2_spec._selector_walk_kernel[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            sample_pos,
            sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.walk_tokens,
            self._selector_scores,
            num_steps=self.num_sample_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=False,
            USE_FP64=self.use_fp64_gumbel,
            num_warps=1,
        )

    @torch.inference_mode()
    def _run_model_with_embeds(
        self,
        num_tokens: int,
        inputs_embeds: torch.Tensor,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            return self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
