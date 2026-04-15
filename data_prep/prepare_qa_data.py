"""
Prepare gene identification puzzle data for NeMo-RL GRPO training.

Source: qa_1000.jsonl — each record contains:
  - clues      Markdown-formatted clues, one per gene (labeled Gene A, Gene B, ...)
  - subgraph   Protein-protein interaction network among the puzzle genes
  - answer_key {label -> HGNC_symbol} mapping
  - n_genes    Number of genes in the puzzle (2-20)

Output JSONL format (one record per line):
  {
    "input":   "<system_prompt + subgraph + clues>",
    "output":  "Gene A: SYMBOL\\nGene B: SYMBOL\\n..."  # sorted by label
  }

A separate system prompt file (prompts/gene_puzzle_system_prompt.txt) is
prepended to every input. The model is expected to reason and then produce
a final answer block in the exact format above.

Usage:
    python data_prep/prepare_qa_data.py \\
        --input          /path/to/qa_1000.jsonl \\
        --system-prompt  prompts/gene_puzzle_system_prompt.txt \\
        --output         qa_train.jsonl \\
        --val-output     qa_val.jsonl \\
        --val-fraction   0.1 \\
        --seed           42
"""

import argparse
import json
import pathlib
import random


def load_system_prompt(path: pathlib.Path) -> str:
    return path.read_text().strip()


def format_input(system_prompt: str, sample: dict) -> str:
    """Combine system prompt, interaction subgraph, and clues into a single user turn."""
    # Strip trailing markdown artefacts from clue generation
    clues = sample["clues"].rstrip().rstrip("*-").rstrip()
    subgraph = sample["subgraph"].strip()
    return (
        f"{system_prompt}\n\n"
        f"## Gene Interaction Network\n\n"
        f"{subgraph}\n\n"
        f"## Clues\n\n"
        f"{clues}"
    )


def format_output(answer_key: dict) -> str:
    """Canonical ground-truth answer block sorted by label."""
    return "\n".join(
        f"{label}: {symbol}"
        for label, symbol in sorted(answer_key.items())
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert qa_1000.jsonl into NeMo-RL JSONL for gene puzzle training."
    )
    parser.add_argument("--input", default="qa_1000.jsonl")
    parser.add_argument("--system-prompt", default="prompts/gene_puzzle_system_prompt.txt")
    parser.add_argument("--output", default="qa_train.jsonl")
    parser.add_argument("--val-output", default="qa_val.jsonl")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction held out for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    system_prompt = load_system_prompt(pathlib.Path(args.system_prompt))
    raw = [
        json.loads(line)
        for line in pathlib.Path(args.input).read_text().strip().splitlines()
    ]
    print(f"Loaded {len(raw)} QA samples  (n_genes range: "
          f"{min(s['n_genes'] for s in raw)}-{max(s['n_genes'] for s in raw)})")

    random.seed(args.seed)
    random.shuffle(raw)

    n_val = max(1, round(len(raw) * args.val_fraction))
    val_raw = raw[:n_val]
    train_raw = raw[n_val:]

    def write_split(samples: list[dict], path: str) -> None:
        with pathlib.Path(path).open("w") as f:
            for s in samples:
                record = {
                    "input": format_input(system_prompt, s),
                    "output": format_output(s["answer_key"]),
                    "n_genes": s["n_genes"],
                }
                f.write(json.dumps(record) + "\n")

    write_split(train_raw, args.output)
    print(f"Wrote {len(train_raw)} training samples -> {args.output}")

    write_split(val_raw, args.val_output)
    print(f"Wrote {len(val_raw)} validation samples -> {args.val_output}")

    ex = train_raw[0]
    print(f"\n--- Sample (n_genes={ex['n_genes']}) ---")
    print("INPUT (first 500 chars):")
    print(format_input(system_prompt, ex)[:500])
    print("\nOUTPUT (ground truth):")
    print(format_output(ex["answer_key"]))


if __name__ == "__main__":
    main()
