# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Gene identification puzzle RL environment for NeMo-RL GRPO training.

Task
----
Given clues describing genes by function, pathway role, and disease
associations, the model must identify each gene by its official HGNC symbol
and output a labelled answer block:

    Gene A: TP53
    Gene B: BRCA1
    ...

The system prompt (see prompts/gene_puzzle_system_prompt.txt) instructs the
model to reason first and then produce the answer block.  The parser uses a
last-wins strategy so the model's final block prevails over any scratchpad
guesses made during reasoning.

Reward
------
Configurable via ``reward_type`` in the environment config:
  - ``"fraction"``  (default) correct / n_genes  — in [0, 1], comparable
                    across puzzles of different sizes
  - ``"count"``     raw integer number of correct identifications

Installation (NeMo-RL)
----------------------
1. Copy this file into ``nemo_rl/environments/``.
2. Add the entry below to the ``ENV_REGISTRY`` dict in
   ``nemo_rl/environments/utils.py``::

       "gene_puzzle": {
           "actor_class_fqn":
               "nemo_rl.environments.gene_puzzle_environment.GenePuzzleEnvironment",
       },

3. Reference ``env_name: gene_puzzle`` in your YAML config (see
   ``configs/grpo_gene_puzzle.yaml``).
"""

import re
from typing import TypedDict

from typing_extensions import NotRequired

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Matches "Gene A: TP53", "gene b: brca1", etc.  Case-insensitive.
_ANSWER_LINE_RE = re.compile(
    r"^\s*Gene\s+([A-Za-z])\s*:\s*([A-Z][A-Z0-9]{1,9})\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_answer_block(text: str) -> dict[str, str]:
    """Extract ``{label -> SYMBOL}`` from model output.

    Scans all matching lines and keeps the **last** occurrence of each label,
    so the final answer block wins over any draft guesses in the reasoning
    scratchpad.

    Returns:
        Dict mapping uppercase label letters ('A', 'B', ...) to uppercase
        gene symbols ('TP53', ...).
    """
    parsed: dict[str, str] = {}
    for m in _ANSWER_LINE_RE.finditer(text):
        parsed[m.group(1).upper()] = m.group(2).upper()
    return parsed


def _parse_ground_truth(output_str: str) -> dict[str, str]:
    """Parse the canonical ``Gene A: SYMBOL\\n...`` ground-truth string."""
    result = {}
    for line in output_str.strip().splitlines():
        m = re.match(r"Gene\s+([A-Za-z])\s*:\s*(\S+)", line, re.IGNORECASE)
        if m:
            result[m.group(1).upper()] = m.group(2).upper()
    return result


def _score_answer(
    predicted: dict[str, str], ground_truth: dict[str, str]
) -> tuple[int, int]:
    """Return ``(num_correct, n_genes)``."""
    n_genes = len(ground_truth)
    correct = sum(
        1
        for label, symbol in ground_truth.items()
        if predicted.get(label, "").upper() == symbol.upper()
    )
    return correct, n_genes


# ---------------------------------------------------------------------------
# Config / metadata types
# ---------------------------------------------------------------------------

class GenePuzzleEnvConfig(TypedDict):
    num_workers: int
    reward_type: NotRequired[str]  # "fraction" | "count"


class GenePuzzleMetadata(TypedDict):
    ground_truth: str  # "Gene A: SYMBOL\nGene B: SYMBOL\n..."


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class GenePuzzleEnvironment(EnvironmentInterface[GenePuzzleMetadata]):
    """Single-turn RL environment: gene clues in, labelled HGNC symbols out.

    Args:
        cfg: Must contain ``num_workers`` (int) and optionally
             ``reward_type`` ("fraction" | "count").
    """

    def __init__(self, cfg: GenePuzzleEnvConfig) -> None:
        self.cfg = cfg
        self.reward_type = cfg.get("reward_type", "fraction")

    # ------------------------------------------------------------------
    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[GenePuzzleMetadata],
    ) -> EnvironmentReturn[GenePuzzleMetadata]:
        rewards = []
        for conversation, meta in zip(message_log_batch, metadata):
            response = "".join(
                str(msg["content"])
                for msg in conversation
                if msg["role"] == "assistant"
            )
            predicted = _parse_answer_block(response)
            ground_truth = _parse_ground_truth(meta["ground_truth"])
            correct, n_genes = _score_answer(predicted, ground_truth)

            if self.reward_type == "count":
                reward = float(correct)
            else:
                reward = correct / n_genes if n_genes > 0 else 0.0

            rewards.append(reward)

        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        done_tensor = torch.ones_like(reward_tensor)
        observations = [
            {"role": "environment", "content": f"Score: {r:.3f}"}
            for r in rewards
        ]
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=reward_tensor,
            terminateds=done_tensor,
            answers=None,
        )

    # ------------------------------------------------------------------
    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        rewards = batch["rewards"]
        if rewards.ndim > 1:
            rewards = rewards[:, 0]
        rewards = rewards * batch["is_end"]

        metrics = {
            "accuracy": rewards.mean().item(),
            "mean_fraction_correct": rewards.mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], rewards
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": rewards.shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
        }
        return batch, metrics
