"""
Eval harness: compare system verdicts against a hand-labeled golden set.

Supports three modes:
- brute: run the brute-force evaluator (every rule x every chunk)
- agent: run the agentic evaluator (targeted tool use)
- compare: run both and print a side-by-side table

This is the centerpiece of the repo. The blog post says an eval harness
with a golden set should have been built first, not last. This module
makes that pattern concrete.

Metrics: per-category precision/recall/F1, overall accuracy, per-rule
agreement table. In compare mode: calls, estimated cost, and F1 across
both evaluators.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from rubric_eval.agent_evaluator import evaluate_document_agentic
from rubric_eval.evaluator import evaluate_document
from rubric_eval.llm import get_llm_client
from rubric_eval.models import CompiledRubric, Document

# Rough cost estimate per LLM call for comparison purposes.
# Based on ~2K input tokens + ~200 output tokens at mid-tier API rates.
# This is an order-of-magnitude estimate, not a precise invoice.
_COST_PER_CALL = 0.003


def _aggregate_verdicts(
    evaluations: list,
) -> dict[str, str]:
    """Aggregate chunk-level verdicts into a per-rule document-level verdict.

    Logic: if ANY chunk returns "pass" for a rule, the document-level
    verdict is "pass." This matches the brute-force pattern.
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
    avg_f1 = round(
        sum(m["f1"] for m in category_metrics.values()) / len(category_metrics), 3
    ) if category_metrics else 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "total_comparisons": total_count,
        "total_matches": total_match,
        "category_metrics": category_metrics,
        "avg_f1": avg_f1,
    }


def _run_single_mode(
    doc_dir: str,
    rubric_path: str,
    provider: str | None,
    mode: str,
) -> list[dict]:
    """Run one evaluator mode on all documents and return per-doc results."""
    client = get_llm_client(provider)

    rubric_data = json.loads(Path(rubric_path).read_text(encoding="utf-8"))
    rubric = CompiledRubric.model_validate(rubric_data)
    rule_categories = {r.rule_id: r.category for r in rubric.rules}

    doc_dir_path = Path(doc_dir)
    doc_files = sorted(doc_dir_path.glob("*.json"))

    results: list[dict] = []

    for doc_file in doc_files:
        doc_data = json.loads(doc_file.read_text(encoding="utf-8"))
        document = Document.model_validate(doc_data)

        if not document.ground_truth:
            continue

        if mode == "agent":
            result = evaluate_document_agentic(document, rubric, client=client)
        else:
            result = evaluate_document(document, rubric, client=client)

        system_verdicts = _aggregate_verdicts(result.evaluations)

        comparisons: list[dict] = []
        for gt in document.ground_truth:
            sys_v = system_verdicts.get(gt.rule_id, "fail")
            category = rule_categories.get(gt.rule_id, "unknown")
            comparisons.append({
                "doc_id": document.doc_id,
                "rule_id": gt.rule_id,
                "category": category,
                "system_verdict": sys_v,
                "ground_truth_verdict": gt.verdict,
                "match": sys_v == gt.verdict,
            })

        metrics = _compute_metrics(comparisons)
        results.append({
            "doc_id": document.doc_id,
            "mode": mode,
            "calls": result.total_llm_calls,
            "accuracy": metrics["overall_accuracy"],
            "avg_f1": metrics["avg_f1"],
            "comparisons": comparisons,
            "category_metrics": metrics["category_metrics"],
        })

    return results


def run_eval(
    doc_dir: str,
    rubric_path: str,
    provider: str | None = None,
    mode: str = "brute",
) -> dict:
    """Run the eval harness in the specified mode.

    Modes:
    - brute: brute-force evaluator (default)
    - agent: agentic evaluator with targeted tool use
    - compare: both, with side-by-side table
    """
    if mode in ("brute", "agent"):
        results = _run_single_mode(doc_dir, rubric_path, provider, mode)
        _print_single_results(results)
        return {"mode": mode, "results": results}

    if mode == "compare":
        brute_results = _run_single_mode(doc_dir, rubric_path, provider, "brute")
        agent_results = _run_single_mode(doc_dir, rubric_path, provider, "agent")
        _print_comparison(brute_results, agent_results)
        return {"mode": "compare", "brute": brute_results, "agent": agent_results}

    raise ValueError(f"Unknown mode: {mode}. Use brute, agent, or compare.")


def _print_single_results(results: list[dict]) -> None:
    """Print formatted metrics for a single mode."""
    print("\n" + "=" * 70)
    print("EVAL HARNESS RESULTS")
    print("=" * 70)

    for r in results:
        print(f"\nDocument: {r['doc_id']} | Calls: {r['calls']} | "
              f"Accuracy: {r['accuracy']} | Avg F1: {r['avg_f1']}")

        print(f"\n{'Category':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4}")
        print("-" * 60)
        for cat, m in sorted(r.get("category_metrics", {}).items()):
            print(
                f"{cat:<25} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                f"{m['f1']:>6.3f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}"
            )

        print(f"\n{'Doc':<12} {'Rule':<14} {'System':<8} {'Truth':<8} {'Match':<6}")
        print("-" * 50)
        for c in r["comparisons"]:
            match_str = "yes" if c["match"] else "NO"
            print(
                f"{c['doc_id']:<12} {c['rule_id']:<14} "
                f"{c['system_verdict']:<8} {c['ground_truth_verdict']:<8} "
                f"{match_str:<6}"
            )

    print("\n" + "=" * 70)


def _print_comparison(brute_results: list[dict], agent_results: list[dict]) -> None:
    """Print a side-by-side comparison of brute-force vs agentic evaluation."""
    print("\n" + "=" * 90)
    print("BRUTE-FORCE VS AGENTIC COMPARISON")
    print("=" * 90)

    print(f"\n{'Document':<14} {'Mode':<8} {'Calls':>8} {'Est. Cost':>12} {'Accuracy':>10} {'Avg F1':>10}")
    print("-" * 90)

    total_brute_calls = 0
    total_agent_calls = 0

    for br, ar in zip(brute_results, agent_results):
        brute_cost = br["calls"] * _COST_PER_CALL
        agent_cost = ar["calls"] * _COST_PER_CALL

        print(
            f"{br['doc_id']:<14} {'brute':<8} {br['calls']:>8} ${brute_cost:>10.3f} "
            f"{br['accuracy']:>10.3f} {br['avg_f1']:>10.3f}"
        )
        print(
            f"{ar['doc_id']:<14} {'agent':<8} {ar['calls']:>8} ${agent_cost:>10.3f} "
            f"{ar['accuracy']:>10.3f} {ar['avg_f1']:>10.3f}"
        )
        print("-" * 90)

        total_brute_calls += br["calls"]
        total_agent_calls += ar["calls"]

    total_brute_cost = total_brute_calls * _COST_PER_CALL
    total_agent_cost = total_agent_calls * _COST_PER_CALL
    savings = total_brute_cost - total_agent_cost
    pct = (savings / total_brute_cost * 100) if total_brute_calls > 0 else 0

    print(
        f"{'TOTAL':<14} {'brute':<8} {total_brute_calls:>8} ${total_brute_cost:>10.3f}"
    )
    print(
        f"{'TOTAL':<14} {'agent':<8} {total_agent_calls:>8} ${total_agent_cost:>10.3f}"
    )
    print(f"\nSavings: ${savings:.3f} ({pct:.1f}% fewer calls)")

    # Per-category F1 comparison
    print("\n" + "-" * 90)
    print("PER-CATEGORY F1 COMPARISON")
    print(f"{'Category':<25} {'Brute F1':>10} {'Agent F1':>10} {'Delta':>10}")
    print("-" * 60)

    # Aggregate category metrics across all docs
    brute_cat_f1: dict[str, list[float]] = defaultdict(list)
    agent_cat_f1: dict[str, list[float]] = defaultdict(list)

    for br in brute_results:
        for cat, m in br.get("category_metrics", {}).items():
            brute_cat_f1[cat].append(m["f1"])
    for ar in agent_results:
        for cat, m in ar.get("category_metrics", {}).items():
            agent_cat_f1[cat].append(m["f1"])

    all_cats = sorted(set(brute_cat_f1.keys()) | set(agent_cat_f1.keys()))
    for cat in all_cats:
        b_avg = round(sum(brute_cat_f1[cat]) / len(brute_cat_f1[cat]), 3) if brute_cat_f1[cat] else 0.0
        a_avg = round(sum(agent_cat_f1[cat]) / len(agent_cat_f1[cat]), 3) if agent_cat_f1[cat] else 0.0
        delta = round(a_avg - b_avg, 3)
        print(f"{cat:<25} {b_avg:>10.3f} {a_avg:>10.3f} {delta:>+10.3f}")

    print("\n" + "=" * 90)


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
    parser.add_argument(
        "--mode",
        choices=["brute", "agent", "compare"],
        default="brute",
        help="Evaluation mode: brute (default), agent, or compare both.",
    )
    args = parser.parse_args()

    run_eval(args.doc_dir, args.rubric_path, provider=args.provider, mode=args.mode)


if __name__ == "__main__":
    main()
