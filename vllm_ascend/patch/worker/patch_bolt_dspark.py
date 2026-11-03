#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""
Patch: BOLT online adaptive speculative decoding for DSpark.

When ``bolt_config.enabled=True``, this patch initializes a ``BoltManager``
on each DSpark draft model instance after construction. The BoltManager holds
per-request delta matrices that are updated online after each verify block
using teacher (target model) feedback, improving draft token acceptance rates.

The patch is a no-op when BOLT is disabled (``bolt_config.enabled=False`` and
``bolt_config.metrics_dir=""``), so it is safe to always import.

Patches:
  - ``Qwen3DSparkForCausalLM.__init__``: initialize BoltManager after super().__init__

Note: only the Qwen3 DSpark lineage (and its subclasses) is patched. Other
DSpark families (Kimi, DeepSeek variants) do not get a ``bolt_manager`` and
BOLT stays inactive for them.
"""

# Use vLLM's configured logger so INFO lines reach the server console;
# a bare logging.getLogger here has no handler and drops INFO silently.
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.spec_decode.bolt_manager import BoltManager


def _patch_bolt_dspark():
    """Patch DSpark model to support BOLT online delta update."""
    try:
        from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM
    except ImportError:
        logger.debug("Qwen3DSparkForCausalLM not available, skipping BOLT patch")
        return

    _original_init = Qwen3DSparkForCausalLM.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)

        ascend_config = get_ascend_config()
        bolt_config = ascend_config.bolt_config

        # Skip if BOLT is not enabled and no metrics requested
        if not bolt_config.enabled and not bolt_config.metrics_dir:
            self.bolt_manager = None
            return

        # Get model config parameters
        r = getattr(self.config, "markov_rank", 256)
        d = getattr(self.config, "hidden_size", 2560)
        K = getattr(self.config, "block_size", 7)
        device = next(self.parameters()).device

        # Initialize BoltManager
        bolt_mgr = BoltManager(
            config=bolt_config, r=r, d=d, K=K, device=device,
        )

        # Set frozen markov head weights
        markov_head = self.model.markov_head
        W1 = markov_head.markov_w1.weight.data
        W2 = markov_head.markov_w2.weight.data
        W0 = None
        b0 = None
        joint_proj = getattr(markov_head, "joint_proj", None)
        if joint_proj is not None:
            W0 = joint_proj.weight.data
            b0 = joint_proj.bias.data
        bolt_mgr.set_weights(W1=W1, W2=W2, W0=W0, b0=b0)

        # Store BoltManager on both the model wrapper and inner model
        self.bolt_manager = bolt_mgr
        self.model.bolt_manager = bolt_mgr

        logger.info(
            "BOLT manager initialized: r=%d, d=%d, K=%d, "
            "optimizer=%s, lr=%s, enabled=%s, device=%s, "
            "head_mode=%s, loss=%s, update_interval=%d",
            r, d, K, bolt_config.optimizer, bolt_config.lr,
            bolt_mgr.enabled, device, bolt_config.head_mode,
            bolt_config.loss, bolt_config.update_interval,
        )

    Qwen3DSparkForCausalLM.__init__ = _patched_init
    logger.info("Patched Qwen3DSparkForCausalLM.__init__ for BOLT support")


_patch_bolt_dspark()