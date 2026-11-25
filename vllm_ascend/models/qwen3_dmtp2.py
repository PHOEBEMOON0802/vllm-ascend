# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""D-MTP2 draft model for the Ascend MRV2 model runner.

Port of the upstream reference implementation (vLLM ``qwen3_dmtp2``): the
DFlash2 trunk (grouped convolutions + candidate selector) plus a frozen
first-token path. The verifier's native MTP block proposes T1 each round and a
zero-initialized residual fusion folds the MTP hidden into T1's slot embedding:

    delta = W_f [RMS_e(e_T1); RMS_m(h_MTP)]
    e~_T1 = e_T1 + delta

The draft block is ``[anchor, T1, MASK x (B-2)]``: slot 0 is the bonus token,
slot 1 carries T1 (proposed by the frozen MTP block before the DFlash2 pass),
slots 2+ are drafted by DFlash2 with the selector walk seeded by T1 as the
first predecessor.

Unlike the earlier hand-rolled attempt, the MTP block is vLLM's own
``Qwen3_5MultiTokenPredictor`` -- the same module the verifier's native MTP
uses -- so it carries the real pre-fc norms, the MoE decoder layer, its own
causal KV cache group, and can run through the regular attention metadata.
"""

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.utils import maybe_prefix

from vllm_ascend.models.qwen3_dflash2 import (
    DFlash2Qwen3ForCausalLM,
    DFlash2Qwen3Model,
)

__all__ = ["DMTP2Qwen3ForCausalLM", "DMTP2Qwen3Model", "MTPTokenFusion"]

# Speculators checkpoints store the frozen MTP block under mtp_layers.0.* with
# training-side module names; Qwen3.5's MTP predictor uses the original HF
# names. (Inverse of speculators' MTPConverter remap.)
_MTP_EXACT_REMAP: dict[str, str] = {
    "mtp_layers.0.input_proj.weight": "fc.weight",
    "mtp_layers.0.final_layernorm.weight": "norm.weight",
}
_MTP_PREFIX_REMAP: list[tuple[str, str]] = [
    ("mtp_layers.0.hidden_layernorm.", "pre_fc_norm_hidden."),
    ("mtp_layers.0.token_layernorm.", "pre_fc_norm_embedding."),
    ("mtp_layers.0.", "layers.0."),
]

_SUPPORTED_MTP_MODEL_TYPES = frozenset(
    {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
)


class MTPTokenFusion(nn.Module):
    """Zero-initialized residual fusion of T1's embedding with its MTP hidden.

    Mirrors speculators' ``MTPTokenFusion``; registered as ``fusion`` so the
    checkpoint's ``fusion.*`` tensors load through the DFlash weight loader
    unchanged.
    """

    def __init__(
        self,
        hidden_size: int,
        rms_norm_eps: float,
        params_dtype: torch.dtype,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.emb_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mtp_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.fusion_proj = ReplicatedLinear(
            2 * hidden_size,
            hidden_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "fusion_proj"),
            return_bias=False,
        )

    def forward(
        self,
        token_embedding: torch.Tensor,  # [..., hidden_size]
        mtp_hidden: torch.Tensor,  # [..., hidden_size]
    ) -> torch.Tensor:
        fused = torch.cat(
            [
                self.emb_norm(token_embedding),
                self.mtp_norm(mtp_hidden.to(token_embedding.dtype)),
            ],
            dim=-1,
        )
        return token_embedding + self.fusion_proj(fused)


class DMTP2Qwen3Model(DFlash2Qwen3Model):
    """DFlash2 draft trunk that also owns the T1 fusion module."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        self.fusion = MTPTokenFusion(
            hidden_size=self.config.hidden_size,
            rms_norm_eps=self.config.rms_norm_eps,
            params_dtype=vllm_config.model_config.dtype,
            prefix=maybe_prefix(prefix, "fusion"),
        )


class DMTP2Qwen3ForCausalLM(DFlash2Qwen3ForCausalLM):
    """D-MTP2 draft model: DFlash2 trunk + frozen-MTP first token."""

    model_cls = DMTP2Qwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # ``vllm_config.model_config`` is still the TARGET's config here: the
        # draft loader only swaps the attention/cache settings, so the frozen
        # MTP block is built from the verifier architecture the draft was
        # trained against.
        target_text_config = vllm_config.model_config.hf_text_config
        if target_text_config.model_type not in _SUPPORTED_MTP_MODEL_TYPES:
            raise NotImplementedError(
                "D-MTP2 builds its frozen MTP block from Qwen3.5-family "
                f"targets; got model_type={target_text_config.model_type!r}."
            )
        mtp_hidden = getattr(self.config, "dflash_config", {}).get("mtp_hidden_size")
        if mtp_hidden is not None and mtp_hidden != target_text_config.hidden_size:
            raise ValueError(
                "The draft's frozen MTP block was trained with hidden_size="
                f"{mtp_hidden} but the target has {target_text_config.hidden_size}; "
                "the target must be the verifier the draft was trained against."
            )

        from vllm.compilation.backends import set_model_tag
        from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor

        with set_model_tag("dmtp2_mtp"):
            self.mtp = Qwen3_5MultiTokenPredictor(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
            )
        self.mtp.requires_grad_(False)
        assert self.draft_id_to_target_id is None, (
            "D-MTP2 requires draft_vocab_size == the target vocab size."
        )

    # -- frozen MTP pass -------------------------------------------------

    def mtp_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """One frozen-MTP pass over the (shifted) target batch.

        Row t carries ``(h_t, emb(x_{t+1}))`` and its output predicts
        ``x_{t+2}``; the last valid row of each request is the hidden that
        proposes T1. Running the whole batch also keeps the MTP KV in sync.
        """
        return self.mtp(input_ids, positions, hidden_states)

    def compute_t1_logits(self, mtp_hidden: torch.Tensor) -> torch.Tensor:
        # The MTP block shares the verifier LM head, which the DFlash loader
        # already shares into self.lm_head.
        return self.logits_processor(self.lm_head, mtp_hidden)

    def fuse_t1_embedding(
        self, t1_embedding: torch.Tensor, mtp_hidden: torch.Tensor
    ) -> torch.Tensor:
        return self.model.fusion(t1_embedding, mtp_hidden)

    # -- draft attention / KV wiring -------------------------------------

    def _mtp_attn_layer_names(self) -> list[str]:
        from vllm.model_executor.layers.attention import Attention

        names = [
            module.layer_name
            for module in self.mtp.modules()
            if isinstance(module, Attention)
        ]
        assert names, "The frozen MTP block registered no attention layers."
        return names

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        # The MTP layers come LAST so the context-KV slot lists built from this
        # order still index the DFlash2 layers 0..L-1.
        return super().get_draft_kv_cache_layer_names() + self._mtp_attn_layer_names()

    def get_draft_attn_causal(self) -> list[bool]:
        causal = super().get_draft_attn_causal()
        causal.extend(True for _ in self._mtp_attn_layer_names())
        return causal

    # -- weights ----------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        trunk_weights: list[tuple[str, torch.Tensor]] = []
        mtp_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if name.startswith("mtp_layers."):
                mtp_weights.append((self._remap_mtp_key(name), weight))
            else:
                # fusion.* flows through the parent loader as model.fusion.*.
                trunk_weights.append((name, weight))
        if mtp_weights:
            self.mtp.load_weights(mtp_weights)
        super().load_weights(trunk_weights)

    @staticmethod
    def _remap_mtp_key(name: str) -> str:
        if name in _MTP_EXACT_REMAP:
            return _MTP_EXACT_REMAP[name]
        for src, dst in _MTP_PREFIX_REMAP:
            if name.startswith(src):
                return dst + name[len(src) :]
        return name


EntryClass = DMTP2Qwen3ForCausalLM
