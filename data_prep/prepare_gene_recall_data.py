"""
Prepare gene recall data for NeMo-RL GRPO training.

Source: textbook_themes_v2/ directory containing paired .narrative and .genes files.
  - .narrative  One paragraph describing a cancer biology process.
  - .genes      One HGNC gene symbol per line (the ground-truth gene set).

Output JSONL format (one record per line):
  {
    "input":  "<full prompt asking model to list gene symbols>",
    "output": "GENE1, GENE2, ..."   # comma-separated sorted ground-truth symbols
  }

The prompt instructs the model to reason first, then emit a final structured
line beginning with "Gene symbols:" so the reward extractor can anchor on it.

Usage:
    python data_prep/prepare_gene_recall_data.py \\
        --themes-dir /path/to/textbook_themes_v2 \\
        --output     gene_recall_train.jsonl \\
        --val-output gene_recall_val.jsonl \\
        --val-fraction 0.1 \\
        --seed 42
"""

import argparse
import json
import pathlib
import random


# Keep in sync with gene_recall_environment.py and eval_gene_recall.py.
# The "Gene symbols:" anchor lets the reward function ignore reasoning prose.
PROMPT_TEMPLATE = (
    "Identify the core genes involved in the following biological narrative. "
    "Think through which genes are directly mentioned or clearly implied as central players. "
    "After your reasoning, output a final line that begins exactly with 'Gene symbols:' "
    "followed by a comma-separated list of HGNC gene symbols (e.g. TP53, BRCA1, MYC) "
    "and nothing else on that line.\n\n"
    "Narrative: {narrative}\n\n"
)


def load_themes(themes_dir: pathlib.Path) -> list[dict]:
    themes = []
    for narrative_path in sorted(themes_dir.glob("*.narrative")):
        genes_path = narrative_path.with_suffix(".genes")
        if not genes_path.exists():
            print(f"[warn] no .genes file for {narrative_path.name}, skipping")
            continue
        narrative = narrative_path.read_text().strip()
        genes = [
            line.strip()
            for line in genes_path.read_text().strip().splitlines()
            if line.strip()
        ]
        themes.append({
            "theme": narrative_path.stem,
            "narrative": narrative,
            "genes": genes,
        })
    return themes


def format_sample(theme: dict) -> dict:
    return {
        "input": PROMPT_TEMPLATE.format(narrative=theme["narrative"]),
        "output": ", ".join(sorted(theme["genes"])),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert textbook_themes_v2 into NeMo-RL JSONL for gene recall training."
    )
    parser.add_argument("--themes-dir", default="textbook_themes_v2",
                        help="Directory containing .narrative and .genes files")
    parser.add_argument("--output", default="gene_recall_train.jsonl")
    parser.add_argument("--val-output", default="gene_recall_val.jsonl")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of themes held out for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    themes_dir = pathlib.Path(args.themes_dir)
    themes = load_themes(themes_dir)
    print(f"Loaded {len(themes)} themes from {themes_dir}")

    random.seed(args.seed)
    random.shuffle(themes)

    n_val = max(1, round(len(themes) * args.val_fraction))
    val_themes = themes[:n_val]
    train_themes = themes[n_val:]

    out_path = pathlib.Path(args.output)
    with out_path.open("w") as f:
        for theme in train_themes:
            f.write(json.dumps(format_sample(theme)) + "\n")
    print(f"Wrote {len(train_themes)} training samples -> {out_path}")

    val_path = pathlib.Path(args.val_output)
    with val_path.open("w") as f:
        for theme in val_themes:
            f.write(json.dumps(format_sample(theme)) + "\n")
    print(f"Wrote {len(val_themes)} validation samples -> {val_path}")

    sample = format_sample(train_themes[0])
    print("\n--- Sample input (truncated) ---")
    print(sample["input"][:400] + "...")
    print("\n--- Sample output (ground truth) ---")
    print(sample["output"])


if __name__ == "__main__":
    main()
