# TPC CINECA RL — Datasets & Reward Functions for NeMo-RL

This repository contains datasets, reward-function environments, and GRPO
training configs for RL fine-tuning of LLMs on cancer biology tasks using
[NeMo-RL](https://github.com/NVIDIA-NeMo/RL).

Two tasks are provided, both using GRPO as the RL algorithm:

| Task | Dataset | Environment | Config |
|---|---|---|---|
| Gene Recall | `textbook_themes_v2/` | `gene_recall_environment.py` | `grpo_gene_recall.yaml` |
| Gene Puzzle | `qa_1000.jsonl` | `gene_puzzle_environment.py` | `grpo_gene_puzzle.yaml` |

---

## Repository layout

```
tpc_cineca_rl/
├── data_prep/
│   ├── prepare_gene_recall_data.py   # convert textbook_themes_v2 → JSONL
│   └── prepare_qa_data.py            # convert qa_1000.jsonl → JSONL
├── environments/
│   ├── gene_recall_environment.py    # F1 reward for free-form gene lists
│   └── gene_puzzle_environment.py    # fraction-correct reward for labelled answers
├── configs/
│   ├── grpo_gene_recall.yaml         # GRPO config for gene recall
│   └── grpo_gene_puzzle.yaml         # GRPO config for gene puzzle
└── prompts/
    └── gene_puzzle_system_prompt.txt # system prompt for the puzzle task
```

---

## Prerequisites

- A working [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) installation
- The raw data files:
  - `textbook_themes_v2/` — directory of `.narrative` / `.genes` pairs
  - `qa_1000.jsonl` — gene identification puzzle records

---

## Step 1 — Prepare the data

### Gene Recall

```bash
python data_prep/prepare_gene_recall_data.py \
    --themes-dir /path/to/textbook_themes_v2 \
    --output     gene_recall_train.jsonl \
    --val-output gene_recall_val.jsonl \
    --val-fraction 0.1 \
    --seed 42
```

Each output record has two keys:

```json
{
  "input":  "Identify the core genes ... Narrative: <paragraph>",
  "output": "BAX, BCL2, CASP3, TP53, ..."
}
```

The prompt instructs the model to reason and then emit a final line beginning
with `Gene symbols:` so the reward extractor can anchor on it rather than
scanning all prose.

### Gene Puzzle

```bash
python data_prep/prepare_qa_data.py \
    --input          /path/to/qa_1000.jsonl \
    --system-prompt  prompts/gene_puzzle_system_prompt.txt \
    --output         qa_train.jsonl \
    --val-output     qa_val.jsonl \
    --val-fraction   0.1 \
    --seed 42
```

Each output record:

```json
{
  "input":  "<system_prompt>\n\n## Gene Interaction Network\n...\n\n## Clues\n...",
  "output": "Gene A: BRCA1\nGene B: TP53\n..."
}
```

---

## Step 2 — Install the environments into NeMo-RL

Copy the environment files into your NeMo-RL checkout:

```bash
cp environments/gene_recall_environment.py  /path/to/NeMo-RL/nemo_rl/environments/
cp environments/gene_puzzle_environment.py  /path/to/NeMo-RL/nemo_rl/environments/
```

Then register them in `nemo_rl/environments/utils.py` by adding two entries
to the `ENV_REGISTRY` dict:

```python
"gene_recall": {
    "actor_class_fqn":
        "nemo_rl.environments.gene_recall_environment.GeneRecallEnvironment",
},
"gene_puzzle": {
    "actor_class_fqn":
        "nemo_rl.environments.gene_puzzle_environment.GenePuzzleEnvironment",
},
```

Add this line to the ACTOR_ENVIRONMENT_REGISTRY in `nemo_rl/distributed/ray_actor_environment_registry.py`

```
"nemo_rl.environments.gene_recall_environment.GeneRecallEnvironment": PY_EXECUTABLES.SYSTEM,
```

---

## Step 3 — Launch GRPO training

Update `data.train.data_path`, `data.validation.data_path`, and
`policy.model_name` in the YAML, or pass them as CLI overrides:

### Gene Recall

```bash
cd /path/to/NeMo-RL
python examples/run_grpo.py \
    --config /path/to/tpc_cineca_rl/configs/grpo_gene_recall.yaml \
    ++data.train.data_path=/path/to/gene_recall_train.jsonl \
    ++data.validation.data_path=/path/to/gene_recall_val.jsonl \
    ++policy.model_name=/path/to/your/model
```

### Gene Puzzle

```bash
cd /path/to/NeMo-RL
python examples/run_grpo.py \
    --config /path/to/tpc_cineca_rl/configs/grpo_gene_puzzle.yaml \
    ++data.train.data_path=/path/to/qa_train.jsonl \
    ++data.validation.data_path=/path/to/qa_val.jsonl \
    ++policy.model_name=/path/to/your/model
```

---

## Adding your own task

To contribute a new dataset + reward function, follow this pattern:

### 1. Prepare the data (`data_prep/prepare_<task>.py`)

Write a script that converts your raw data to JSONL with `input` and `output`
keys:

```python
{"input": "<full prompt text>",  "output": "<ground-truth answer>"}
```

`output` becomes the `ground_truth` field in environment metadata and is
compared against the model's generated response to compute the reward.

### 2. Write the environment (`environments/<task>_environment.py`)

Subclass `EnvironmentInterface` and implement two methods:

```python
import ray, torch
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

@ray.remote(max_restarts=-1, max_task_retries=-1)
class MyEnvironment(EnvironmentInterface):

    def step(self, message_log_batch, metadata):
        rewards = []
        for conversation, meta in zip(message_log_batch, metadata):
            response = "".join(
                str(msg["content"])
                for msg in conversation
                if msg["role"] == "assistant"
            )
            ground_truth = meta["ground_truth"]  # the "output" field from JSONL
            reward = my_scoring_function(response, ground_truth)
            rewards.append(reward)

        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        done_tensor   = torch.ones_like(reward_tensor)
        observations  = [{"role": "environment", "content": ""} for _ in rewards]
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=reward_tensor,
            terminateds=done_tensor,
            answers=None,
        )

    def global_post_process_and_metrics(self, batch):
        rewards = batch["rewards"] * batch["is_end"]
        metrics = {"accuracy": rewards.mean().item()}
        return batch, metrics
```

Key points:
- `metadata[i]["ground_truth"]` holds the `output` string from your JSONL.
- Rewards must be a 1-D float tensor of shape `[batch_size]`.
- `terminateds` should be all-ones for single-turn tasks.
- Return `accuracy` from `global_post_process_and_metrics` — it is the metric
  NeMo-RL checkpoints on (`val:accuracy`).

### 3. Register the environment

Add to `ENV_REGISTRY` in `nemo_rl/environments/utils.py`:

```python
"my_env": {
    "actor_class_fqn": "nemo_rl.environments.my_environment.MyEnvironment",
},
```

### 4. Write the YAML config

Copy `configs/grpo_gene_recall.yaml` as a starting point. Key fields to
adjust for your task:

```yaml
data:
  train:
    dataset_name: "ResponseDataset"
    data_path: "/path/to/train.jsonl"
    input_key: "input"
    output_key: "output"
  validation:
    dataset_name: "ResponseDataset"
    data_path: "/path/to/val.jsonl"
    input_key: "input"
    output_key: "output"
  default:
    env_name: "my_env"        # must match the key in ENV_REGISTRY

env:
  my_env:
    num_workers: 4            # Ray workers for parallel reward computation
    # any other config fields your environment's __init__ reads from cfg

policy:
  max_total_sequence_length: 1024   # set based on your prompt + response length
  generation:
    max_new_tokens: 256             # set based on expected response length

checkpointing:
  metric_name: "val:accuracy"       # must match a key returned by global_post_process_and_metrics
```

---

## Reward design notes

### Gene Recall (`gene_recall_environment.py`)

- Extracts genes by anchoring on `Gene symbols: ...` if present, otherwise
  falls back to scanning the full response with a regex.
- Configurable metric: `f1` (default), `recall`, or `jaccard`.
- F1 balances precision and recall — penalises both missing genes and
  hallucinated ones.

### Gene Puzzle (`gene_puzzle_environment.py`)

- Parses `Gene A: SYMBOL` lines with last-wins semantics, so the model can
  revise answers in its scratchpad before the final block.
- Reward is `correct / n_genes` (fraction), normalised so 2-gene and 20-gene
  puzzles contribute equally to the GRPO gradient.
- Use `reward_type: count` if you prefer the raw integer.
