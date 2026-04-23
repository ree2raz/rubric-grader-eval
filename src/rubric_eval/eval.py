"""
Eval harness: compare system verdicts against a hand-labeled golden set.

This is the centerpiece of the repo. The blog post says an eval harness
with a golden set should have been built first, not last. This module
makes that pattern concrete: load documents with ground truth, run the
evaluator, compare results, print metrics.

Metrics: per-category precision/recall/F1, overall accuracy, per-rule
agreement table.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from rubric_eval.evaluator import evaluate_document
from rubric_eval.llm import get_llm_client
from rubric_eval.models import CompiledRubric, Document


def _aggregate_verdicts(
    evaluations: list,
) -> dict[str, str]:
    """Aggregate chunk-level verdicts into a per-rule document-level verdict.

    Logic: if ANY chunk returns "pass" for a rule, the document-level
    verdict is "pass." This matches the brute-force pattern — the rule
    only needs to be satisfied somewhere in the document.
    """
    rule_verdicts: dict[str, str] = {}
    for ev in evaluations:
        if ev.verdict == "pass":
            rule_verdicts[ev.rule_id] = "pass"
        elif ev.rule_id not in rule_verdicts:
            rule_verdicts[ev.rule_id] = "fail"
    return rule_verdicts


def _compute_metrics(
    comparisons: list[dict],
) -> dict:
    """Compute precision, recall, F1 per category and overall accuracy."""
    # Per-category stats
    categories: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    )
    total_match = 0
    total_count = 0

    for c in comparisons:
        cat = c["category"]
        sys_v = c["system_verdict"]
        gt_v = c["ground_truth_verdict"]
        match = sys_v == gt_v

        if match:
            total_match += 1
        total_count += 1

        # Treating "fail" as positive (the thing we want to detect)
        if gt_v == "fail" and sys_v == "fail":
            categories[cat]["tp"] += 1
        elif gt_v == "pass" and sys_v == "fail":
            categories[cat]["fp"] += 1
        elif gt_v == "fail" and sys_v == "pass":
            categories[cat]["fn"] += 1
        else:
            categories[cat]["tn"] += 1

    category_metrics = {}
    for cat, stats in sorted(categories.items()):
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        category_metrics[cat] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    overall_accuracy = round(total_match / total_count, 3) if total_count > 0 else 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "total_comparisons": total_count,
        "total_matches": total_match,
        "category_metrics": category_metrics,
    }


def run_eval(
    doc_dir: str,
    rubric_path: str,
    provider: str | None = None,
) -> dict:
    """Run the full eval harness.

    1. Load all documents with ground truth from doc_dir
    2. Run evaluator on each
    3. Compare system verdicts to ground truth
    4. Compute and print metrics
    """
    client = get_llm_client(provider)

    # Load rubric
    rubric_data = json.loads(Path(rubric_path).read_text(encoding="utf-8"))
    rubric = CompiledRubric.model_validate(rubric_data)

    # Build a rule_id -> category lookup
    rule_categories = {r.rule_id: r.category for r in rubric.rules}

    # Load documents
    doc_dir_path = Path(doc_dir)
    doc_files = sorted(doc_dir_path.glob("*.json"))
    if not doc_files:
        print(f"No JSON files found in {doc_dir}", file=sys.stderr)
        return {}

    all_comparisons: list[dict] = []
    agreement_rows: list[dict] = []

    for doc_file in doc_files:
        doc_data = json.loads(doc_file.read_text(encoding="utf-8"))
        document = Document.model_validate(doc_data)

        if not document.ground_truth:
            print(
                f"Skipping {document.doc_id}: no ground truth labels",
                file=sys.stderr,
            )
            continue

        # Run evaluator
        result = evaluate_document(document, rubric, client=client)

        # Aggregate chunk verdicts to document level
        system_verdicts = _aggregate_verdicts(result.evaluations)

        # Compare against ground truth
        for gt in document.ground_truth:
            sys_v = system_verdicts.get(gt.rule_id, "fail")
            match = sys_v == gt.verdict
            category = rule_categories.get(gt.rule_id, "unknown")

            comparison = {
                "doc_id": document.doc_id,
                "rule_id": gt.rule_id,
                "category": category,
                "system_verdict": sys_v,
                "ground_truth_verdict": gt.verdict,
                "match": match,
                "severity": gt.severity,
            }
            all_comparisons.append(comparison)
            agreement_rows.append(comparison)

    # Compute metrics
    metrics = _compute_metrics(all_comparisons)

    # Print results
    _print_results(metrics, agreement_rows)

    return metrics


def _print_results(metrics: dict, agreement_rows: list[dict]) -> None:
    """Print formatted metrics tables to stdout."""
    print("\n" + "=" * 70)
    print("EVAL HARNESS RESULTS")
    print("=" * 70)

    # Overall accuracy
    print(
        f"\nOverall accuracy: {metrics['overall_accuracy']} "
        f"({metrics['total_matches']}/{metrics['total_comparisons']})"
    )

    # Per-category metrics
    print(f"\n{'Category':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 60)
    for cat, m in sorted(metrics.get("category_metrics", {}).items()):
        print(
            f"{cat:<25} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}"
        )

    # Per-rule agreement table
    print(f"\n{'Doc':<12} {'Rule':<14} {'System':<8} {'Truth':<8} {'Match':<6}")
    print("-" * 50)
    for row in agreement_rows:
        match_str = "yes" if row["match"] else "NO"
        print(
            f"{row['doc_id']:<12} {row['rule_id']:<14} "
            f"{row['system_verdict']:<8} {row['ground_truth_verdict']:<8} "
            f"{match_str:<6}"
        )

    print("\n" + "=" * 70)


def main():
    """CLI: run eval harness on a set of documents."""
    parser = argparse.ArgumentParser(
        description="Run eval harness: compare system verdicts to golden set."
    )
    parser.add_argument("doc_dir", help="Directory of document JSONs with ground truth.")
    parser.add_argument("rubric_path", help="Path to compiled rubric JSON.")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "vllm"],
        default=None,
        help="LLM provider.",
    )
    args = parser.parse_args()

    run_eval(args.doc_dir, args.rubric_path, provider=args.provider)


if __name__ == "__main__":
    main()
