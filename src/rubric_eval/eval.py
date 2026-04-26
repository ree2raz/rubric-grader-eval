"""
Eval harness: compare system verdicts against a hand-labeled golden set.

Runs the brute-force evaluator against golden-set documents, then computes
per-category precision/recall/F1 and a per-rule agreement table. Supports
an optional --pre-evaluated flag that loads pre-computed verdicts from disk,
skipping LLM calls entirely — the harness becomes a pure metrics calculator.

This is the centerpiece of the repo. The blog post says an eval harness
with a golden set should have been built first, not last. This module
makes that pattern concrete.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from rubric_eval.evaluator import evaluate_document
from rubric_eval.llm import get_llm_client
from rubric_eval.models import CompiledRubric, Document, DocumentEvaluation


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
    """Compute macro-averaged precision, recall, F1 per category and overall accuracy.

    Macro F1 averages the F1 score for both classes (pass and fail) so that
    a document where all ground truth is "pass" and the system agrees does not
    produce a misleading 0.0 F1.
    """
    # Counts for fail-as-positive (what the system catches) and pass-as-positive
    CatCounts = dict[str, int]
    fail_counts: dict[str, CatCounts] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    )
    total_match = 0
    total_count = 0

    for c in comparisons:
        cat = c["category"]
        sys_v = c["system_verdict"]
        gt_v = c["ground_truth_verdict"]

        if sys_v == gt_v:
            total_match += 1
        total_count += 1

        if gt_v == "fail" and sys_v == "fail":
            fail_counts[cat]["tp"] += 1
        elif gt_v == "pass" and sys_v == "fail":
            fail_counts[cat]["fp"] += 1
        elif gt_v == "fail" and sys_v == "pass":
            fail_counts[cat]["fn"] += 1
        else:
            fail_counts[cat]["tn"] += 1

    def _f1(tp: int, fp: int, fn: int) -> float:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    category_metrics = {}
    for cat, fc in sorted(fail_counts.items()):
        fail_tp, fail_fp, fail_fn = fc["tp"], fc["fp"], fc["fn"]
        fail_f1 = _f1(fail_tp, fail_fp, fail_fn)

        # Pass-as-positive: flip the confusion matrix
        pass_tp = fc["tn"]  # both correct on pass
        pass_fp = fc["fn"]  # system pass, truth fail
        pass_fn = fc["fp"]  # system fail, truth pass
        pass_f1 = _f1(pass_tp, pass_fp, pass_fn)

        macro_f1 = round((fail_f1 + pass_f1) / 2.0, 3) if (fail_f1 + pass_f1) > 0 else 0.0

        category_metrics[cat] = {
            "macro_f1": macro_f1,
            "fail_f1": round(fail_f1, 3),
            "pass_f1": round(pass_f1, 3),
            "tp": fail_tp,
            "fp": fail_fp,
            "fn": fail_fn,
        }

    overall_accuracy = round(total_match / total_count, 3) if total_count > 0 else 0.0
    avg_f1 = round(
        sum(m["macro_f1"] for m in category_metrics.values()) / len(category_metrics), 3
    ) if category_metrics else 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "total_comparisons": total_count,
        "total_matches": total_match,
        "category_metrics": category_metrics,
        "avg_f1": avg_f1,
    }


def _evaluate_with_llm(
    doc_dir: str,
    rubric_path: str,
    provider: str | None,
) -> list[dict]:
    """Run the brute-force evaluator against all golden-set documents."""
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
            "calls": result.total_llm_calls,
            "accuracy": metrics["overall_accuracy"],
            "avg_f1": metrics["avg_f1"],
            "comparisons": comparisons,
            "category_metrics": metrics["category_metrics"],
        })

    return results


def _evaluate_from_precomputed(
    doc_dir: str,
    rubric_path: str,
    pre_evaluated_dir: str,
) -> list[dict]:
    """Load pre-computed evaluation JSONs and compare against ground truth.

    No LLM calls are made. The eval harness works as a pure metrics calculator.
    """
    rubric_data = json.loads(Path(rubric_path).read_text(encoding="utf-8"))
    rubric = CompiledRubric.model_validate(rubric_data)
    rule_categories = {r.rule_id: r.category for r in rubric.rules}

    doc_dir_path = Path(doc_dir)
    doc_files = sorted(doc_dir_path.glob("*.json"))
    pre_eval_path = Path(pre_evaluated_dir)

    results: list[dict] = []

    for doc_file in doc_files:
        doc_data = json.loads(doc_file.read_text(encoding="utf-8"))
        document = Document.model_validate(doc_data)

        if not document.ground_truth:
            continue

        # Load pre-computed evaluation for this document
        eval_file = pre_eval_path / doc_file.name
        if not eval_file.exists():
            print(
                f"  Warning: no pre-evaluated file for {document.doc_id} "
                f"(expected {eval_file}), skipping",
                file=sys.stderr,
            )
            continue

        eval_data = json.loads(eval_file.read_text(encoding="utf-8"))
        eval_result = DocumentEvaluation.model_validate(eval_data)

        system_verdicts = _aggregate_verdicts(eval_result.evaluations)

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
            "calls": 0,
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
    pre_evaluated: str | None = None,
) -> dict:
    """Run the eval harness: brute-force evaluation against a golden set.

    With --pre-evaluated, loads verdicts from disk instead of calling LLMs.
    The harness becomes a pure metrics calculator — no API key required.
    """
    if pre_evaluated:
        results = _evaluate_from_precomputed(doc_dir, rubric_path, pre_evaluated)
    else:
        results = _evaluate_with_llm(doc_dir, rubric_path, provider)

    _print_results(results)
    return {"results": results}


def _print_results(results: list[dict]) -> None:
    """Print formatted per-category and per-rule metrics."""
    print("\n" + "=" * 70)
    print("EVAL HARNESS RESULTS")
    print("=" * 70)

    for r in results:
        print(
            f"\nDocument: {r['doc_id']} | Calls: {r['calls']} | "
            f"Accuracy: {r['accuracy']} | Avg F1: {r['avg_f1']}"
        )

        print(
            f"\n{'Category':<25} {'MacroF1':>8} {'FailF1':>8} {'PassF1':>8} {'TP':>4} {'FP':>4} {'FN':>4}"
        )
        print("-" * 60)
        for cat, m in sorted(r.get("category_metrics", {}).items()):
            print(
                f"{cat:<25} {m['macro_f1']:>8.3f} {m['fail_f1']:>8.3f} "
                f"{m['pass_f1']:>8.3f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}"
            )

        print(
            f"\n{'Doc':<12} {'Rule':<14} {'System':<8} {'Truth':<8} {'Match':<6}"
        )
        print("-" * 50)
        for c in r["comparisons"]:
            match_str = "yes" if c["match"] else "NO"
            print(
                f"{c['doc_id']:<12} {c['rule_id']:<14} "
                f"{c['system_verdict']:<8} {c['ground_truth_verdict']:<8} "
                f"{match_str:<6}"
            )

    print("\n" + "=" * 70)


def main():
    """CLI: run eval harness on a set of documents."""
    parser = argparse.ArgumentParser(
        description="Run eval harness: compare system verdicts to golden set."
    )
    parser.add_argument(
        "doc_dir", help="Directory of document JSONs with ground truth."
    )
    parser.add_argument("rubric_path", help="Path to compiled rubric JSON.")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "vllm"],
        default=None,
        help="LLM provider.",
    )
    parser.add_argument(
        "--pre-evaluated",
        default=None,
        help="Directory of pre-computed evaluation JSONs. When set, no LLM calls are made.",
    )
    args = parser.parse_args()

    run_eval(
        args.doc_dir,
        args.rubric_path,
        provider=args.provider,
        pre_evaluated=args.pre_evaluated,
    )


if __name__ == "__main__":
    main()
