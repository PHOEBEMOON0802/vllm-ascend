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

"""Teach the pinned vLLM about D-MTP2 (dmtp2) draft checkpoints.

Port of the upstream reference's ``algos.update_dmtp2``. The pinned vLLM's
speculators bridge only knows dflash/dflash2/dspark/eagle3/peagle, and its
``SpeculativeMethod`` has no ``"dmtp2"`` entry, so a DMTP2 checkpoint would
fail to load. Two adaptations here:

1. Register ``dmtp2`` in ``SUPPORTED_SPECULATORS_TYPES`` and translate the
   checkpoint exactly like the reference does (DFlash2 mapping + the
   ``DMTP2DraftModel`` architecture + ``mtp_hidden_size``).
2. Map the resulting method onto ``"dflash"``, the same way dflash2 is
   handled. Runtime dispatch still keys off the draft architectures, in
   ``vllm_ascend.worker.v2.spec_decode.init_speculator``.

No extra block-drop patch is needed: this vLLM keeps the trailing-block drop
for every ``use_eagle()`` method (``use_eagle_block_drop`` is
``use_eagle() and not disable_eagle_block_drop``, and ``use_eagle()`` covers
``dflash``), which is what D-MTP2's EAGLE-style MTP KV requires.
"""

from vllm.transformers_utils.configs.speculators.algos import (
    SUPPORTED_SPECULATORS_TYPES,
    update_dflash2,
)
from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

DMTP2_ARCHITECTURE = "DMTP2DraftModel"
DMTP2_SPECULATORS_TYPE = "dmtp2"


def update_dmtp2(config_dict: dict, pre_trained_config: dict) -> None:
    """Translate a speculators ``dmtp2`` config into a vLLM draft config."""
    update_dflash2(config_dict=config_dict, pre_trained_config=pre_trained_config)
    pre_trained_config["architectures"] = [DMTP2_ARCHITECTURE]

    # The runtime rebuilds the MTP block from the target's own config; the
    # checkpoint value is kept only so the model can validate the pairing.
    mtp_tc = config_dict.get("mtp_transformer_config") or {}
    mtp_hidden = mtp_tc.get("hidden_size")
    if mtp_hidden is not None:
        pre_trained_config["dflash_config"]["mtp_hidden_size"] = mtp_hidden


SUPPORTED_SPECULATORS_TYPES[DMTP2_SPECULATORS_TYPE] = update_dmtp2


_orig_build_vllm_speculative_config = (
    SpeculatorsConfig.build_vllm_speculative_config.__func__
)


def _build_vllm_speculative_config(cls, config_dict: dict) -> dict:
    result = _orig_build_vllm_speculative_config(cls, config_dict=config_dict)
    if result.get("method") == DMTP2_SPECULATORS_TYPE:
        result["method"] = "dflash"
    return result


SpeculatorsConfig.build_vllm_speculative_config = classmethod(
    _build_vllm_speculative_config
)
