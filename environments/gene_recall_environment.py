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

"""Gene recall RL environment for NeMo-RL GRPO training.

Task
----
Given a narrative describing a cancer biology process, the model must identify
the core genes involved and output them as a comma-separated HGNC symbol list.

The prompt (see data_prep/prepare_gene_recall_data.py) instructs the model to
reason first, then end with an anchored line:

    Gene symbols: TP53, BCL2, BAX, ...

The extractor anchors on this line so reasoning prose does not pollute the
gene set.  If no anchor line is present the full response is scanned as a
fallback.

Reward
------
Configurable via ``reward_metric`` in the environment config:
  - ``"f1"``      (default) F1 between predicted and reference sets
  - ``"recall"``  TP / len(reference)
  - ``"jaccard"`` TP / |predicted ∪ reference|

Installation (NeMo-RL)
----------------------
1. Copy this file into ``nemo_rl/environments/``.
2. Add the entry below to the ``ENV_REGISTRY`` dict in
   ``nemo_rl/environments/utils.py``::

       "gene_recall": {
           "actor_class_fqn":
               "nemo_rl.environments.gene_recall_environment.GeneRecallEnvironment",
       },

3. Reference ``env_name: gene_recall`` in your YAML config (see
   ``configs/grpo_gene_recall.yaml``).
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
# Gene symbol extraction
# ---------------------------------------------------------------------------

# Common English words / abbreviations that match the gene-symbol regex but
# are not gene names.  Extend as needed.
_STOP_WORDS = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN",
    "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM",
    "HIS", "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE", "TWO",
    "WAY", "WHO", "BOY", "DID", "HIT", "LET", "MEN", "PUT", "SAY",
    "SHE", "TOO", "USE", "LIST", "CORE", "GENE", "GENES", "INVOLVED",
    "FOLLOWING", "NARRATIVE", "INCLUDE", "INCLUDING", "SUCH", "AS",
    "IN", "OF", "TO", "IS", "IT", "BE", "AT", "BY", "AN", "OR",
    "IF", "NO", "UP", "SO", "DO", "GO", "ME", "MY", "ON", "WE",
    "DNA", "RNA", "ATP", "ADP", "ECM", "ROS", "TGF", "EGF",
}

_GENE_SYMBOL_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")
_ANCHOR_RE = re.compile(r"(?i)gene\s+symbols\s*:")


def _extract_genes_from_response(text: str) -> set[str]:
    """Extract HGNC gene symbols from the structured ``Gene symbols: ...`` anchor line.

    Returns an empty set if the anchor is absent, so responses that echo the
    prompt or omit the required format receive zero reward.
    """
    for line in text.splitlines():
        if _ANCHOR_RE.match(line):
            candidates = _GENE_SYMBOL_RE.findall(line.split(":", 1)[1])
            return {g for g in candidates if g not in _STOP_WORDS}
    return set()


def _extract_genes_from_reference(text: str) -> set[str]:
    """Extract HGNC gene symbols from a plain comma-separated ground-truth string."""
    candidates = _GENE_SYMBOL_RE.findall(text)
    return {g for g in candidates if g not in _STOP_WORDS}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _f1(predicted: set[str], reference: set[str]) -> float:
    if not predicted and not reference:
        return 1.0
    if not predicted or not reference:
        return 0.0
    tp = len(predicted & reference)
    precision = tp / len(predicted)
    recall = tp / len(reference)
    denom = precision + recall
    return 2 * precision * recall / denom if denom else 0.0


# ---------------------------------------------------------------------------
# Config / metadata types
# ---------------------------------------------------------------------------

class GeneRecallEnvConfig(TypedDict):
    num_workers: int
    reward_metric: NotRequired[str]  # "f1" | "recall" | "jaccard"


class GeneRecallMetadata(TypedDict):
    ground_truth: str  # comma-separated reference gene symbols


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class GeneRecallEnvironment(EnvironmentInterface[GeneRecallMetadata]):
    """Single-turn RL environment: narrative in, gene list out.

    Args:
        cfg: Must contain ``num_workers`` (int) and optionally
             ``reward_metric`` ("f1" | "recall" | "jaccard").
    """

    def __init__(self, cfg: GeneRecallEnvConfig) -> None:
        self.cfg = cfg
        self.reward_metric = cfg.get("reward_metric", "f1")

    # ------------------------------------------------------------------
    def _score(self, predicted: set[str], reference: set[str]) -> float:
        if self.reward_metric == "recall":
            return len(predicted & reference) / len(reference) if reference else 1.0
        if self.reward_metric == "jaccard":
            union = predicted | reference
            return len(predicted & reference) / len(union) if union else 1.0
        return _f1(predicted, reference)

    # ------------------------------------------------------------------
    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[GeneRecallMetadata],
    ) -> EnvironmentReturn[GeneRecallMetadata]:
        rewards = []
        for conversation, meta in zip(message_log_batch, metadata):
            response = "".join(
                str(msg["content"])
                for msg in conversation
                if msg["role"] == "assistant"
            )
            predicted = _extract_genes_from_response(response)
            reference = _extract_genes_from_reference(meta["ground_truth"])
            rewards.append(self._score(predicted, reference))

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
            f"mean_{self.reward_metric}": rewards.mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], rewards
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": rewards.shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
        }
        return batch, metrics
