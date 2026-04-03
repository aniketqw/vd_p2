"""Extract misclassified sample info into a standalone JSON file.

Reads a training log JSON (written by `DebugLogger`) and writes a new JSON
containing only the misclassified sample information, optionally capped at a
maximum number of entries.

Example:
    python export_misclassified.py \\
      --input  logs/training_log_20260317_005331.json \\
      --output logs/misclassified_20260317_005331.json
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import argparse
import datetime
import json


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract misclassified samples from a DebugLogger training log.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",  required=True,
        help="Path to the training log JSON produced by DebugLogger",
    )
    parser.add_argument(
        "--output", required=True,
        help="Destination path for the extracted misclassified JSON",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="If set, only include the first N misclassified samples",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    misclassified = data.get("misclassified_samples", [])
    if args.max_samples is not None:
        misclassified = misclassified[: args.max_samples]

    output = {
        "source_log":          args.input,
        "created_at":          datetime.datetime.now().isoformat(),
        "n_misclassified":     len(misclassified),
        "misclassified_samples": misclassified,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(misclassified)} misclassified samples → {args.output}")


if __name__ == "__main__":
    main()
