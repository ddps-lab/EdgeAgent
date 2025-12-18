"""Compare classification results across different methods and models."""

import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import Counter
import csv

from .models import ClassificationResult, load_results_from_csv
from .config import RESULTS_DIR


def calculate_agreement_rate(
    results1: List[ClassificationResult],
    results2: List[ClassificationResult],
    level: str = "major",
) -> Dict:
    """
    Calculate agreement rate between two classification results.

    Args:
        results1: First set of results
        results2: Second set of results
        level: "major" or "minor" category level

    Returns:
        Dictionary with agreement statistics
    """
    # Create lookup by server_id
    lookup1 = {r.server_id: r for r in results1}
    lookup2 = {r.server_id: r for r in results2}

    # Find common servers
    common_ids = set(lookup1.keys()) & set(lookup2.keys())

    if not common_ids:
        return {"agreement_rate": 0, "total_compared": 0}

    agreements = 0
    disagreements = []

    for server_id in common_ids:
        r1 = lookup1[server_id]
        r2 = lookup2[server_id]

        if level == "major":
            match = r1.category_major == r2.category_major
        else:
            match = (r1.category_major == r2.category_major and
                     r1.category_minor == r2.category_minor)

        if match:
            agreements += 1
        else:
            disagreements.append({
                "server_id": server_id,
                "server_name": r1.server_name,
                "result1": {
                    "major": r1.category_major,
                    "minor": r1.category_minor,
                    "method": r1.method,
                },
                "result2": {
                    "major": r2.category_major,
                    "minor": r2.category_minor,
                    "method": r2.method,
                },
            })

    return {
        "agreement_rate": agreements / len(common_ids),
        "total_compared": len(common_ids),
        "agreements": agreements,
        "disagreements_count": len(disagreements),
        "disagreements_sample": disagreements[:20],  # Sample of disagreements
    }


def calculate_cohens_kappa(
    results1: List[ClassificationResult],
    results2: List[ClassificationResult],
    level: str = "major",
) -> float:
    """
    Calculate Cohen's Kappa for inter-rater reliability.

    Args:
        results1: First set of results
        results2: Second set of results
        level: "major" or "minor" category level

    Returns:
        Cohen's Kappa coefficient (-1 to 1, where 1 is perfect agreement)
    """
    lookup1 = {r.server_id: r for r in results1}
    lookup2 = {r.server_id: r for r in results2}
    common_ids = set(lookup1.keys()) & set(lookup2.keys())

    if not common_ids:
        return 0.0

    # Get categories
    categories1 = []
    categories2 = []

    for server_id in common_ids:
        r1 = lookup1[server_id]
        r2 = lookup2[server_id]

        if level == "major":
            categories1.append(r1.category_major)
            categories2.append(r2.category_major)
        else:
            categories1.append(f"{r1.category_major}/{r1.category_minor}")
            categories2.append(f"{r2.category_major}/{r2.category_minor}")

    # Calculate observed agreement
    n = len(common_ids)
    observed = sum(1 for c1, c2 in zip(categories1, categories2) if c1 == c2) / n

    # Calculate expected agreement
    counter1 = Counter(categories1)
    counter2 = Counter(categories2)
    all_categories = set(counter1.keys()) | set(counter2.keys())

    expected = sum(
        (counter1.get(cat, 0) / n) * (counter2.get(cat, 0) / n)
        for cat in all_categories
    )

    # Cohen's Kappa
    if expected == 1:
        return 1.0 if observed == 1 else 0.0

    kappa = (observed - expected) / (1 - expected)
    return kappa


def compare_distributions(
    results_dict: Dict[str, List[ClassificationResult]],
    level: str = "major",
) -> Dict:
    """
    Compare category distributions across different methods/models.

    Args:
        results_dict: Dictionary mapping method name to results
        level: "major" or "minor" category level

    Returns:
        Distribution comparison statistics
    """
    distributions = {}

    for method_name, results in results_dict.items():
        if not results:
            continue

        if level == "major":
            categories = [r.category_major for r in results]
        else:
            categories = [f"{r.category_major}/{r.category_minor}" for r in results]

        counter = Counter(categories)
        total = len(categories)

        distributions[method_name] = {
            cat: {"count": count, "percentage": count / total * 100}
            for cat, count in counter.most_common()
        }

    return distributions


def compare_confidence_scores(
    results_dict: Dict[str, List[ClassificationResult]],
) -> Dict:
    """
    Compare confidence scores across different methods/models.

    Args:
        results_dict: Dictionary mapping method name to results

    Returns:
        Confidence statistics by method
    """
    stats = {}

    for method_name, results in results_dict.items():
        if not results:
            continue

        confidences = [r.confidence for r in results]

        stats[method_name] = {
            "mean": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "std": (sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)) ** 0.5,
            "count": len(confidences),
        }

    return stats


def generate_comparison_report(
    results_dict: Dict[str, List[ClassificationResult]],
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Generate a comprehensive comparison report.

    Args:
        results_dict: Dictionary mapping method name to results
        output_dir: Directory to save report files

    Returns:
        Comprehensive comparison report
    """
    output_dir = output_dir or RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "summary": {},
        "pairwise_agreement": {},
        "cohens_kappa": {},
        "distributions": {},
        "confidence_stats": {},
    }

    # Summary
    for method, results in results_dict.items():
        report["summary"][method] = {
            "total_classified": len(results),
            "unique_major_categories": len(set(r.category_major for r in results)) if results else 0,
        }

    # Pairwise agreement (major level)
    methods = list(results_dict.keys())
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            pair_name = f"{method1}_vs_{method2}"
            agreement = calculate_agreement_rate(
                results_dict[method1],
                results_dict[method2],
                level="major"
            )
            report["pairwise_agreement"][pair_name] = agreement

            # Cohen's Kappa
            kappa = calculate_cohens_kappa(
                results_dict[method1],
                results_dict[method2],
                level="major"
            )
            report["cohens_kappa"][pair_name] = kappa

    # Distributions
    report["distributions"] = compare_distributions(results_dict, level="major")

    # Confidence
    report["confidence_stats"] = compare_confidence_scores(results_dict)

    # Save JSON report
    json_path = output_dir / "comparison_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Generate Markdown report
    md_report = generate_markdown_report(report)
    md_path = output_dir / "comparison_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)

    print(f"Comparison report saved to:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")

    return report


def generate_markdown_report(report: Dict) -> str:
    """Generate a human-readable Markdown report."""
    lines = ["# MCP Tool Classification Comparison Report\n"]

    # Summary
    lines.append("## Summary\n")
    lines.append("| Method | Total Classified | Unique Categories |")
    lines.append("|--------|-----------------|-------------------|")
    for method, stats in report["summary"].items():
        lines.append(f"| {method} | {stats['total_classified']} | {stats['unique_major_categories']} |")
    lines.append("")

    # Pairwise Agreement
    lines.append("## Pairwise Agreement (Major Category)\n")
    lines.append("| Comparison | Agreement Rate | Cohen's Kappa |")
    lines.append("|------------|----------------|---------------|")
    for pair_name, agreement in report["pairwise_agreement"].items():
        kappa = report["cohens_kappa"].get(pair_name, 0)
        rate = agreement.get("agreement_rate", 0)
        lines.append(f"| {pair_name} | {rate:.2%} | {kappa:.3f} |")
    lines.append("")

    # Confidence Statistics
    lines.append("## Confidence Statistics\n")
    lines.append("| Method | Mean | Std | Min | Max |")
    lines.append("|--------|------|-----|-----|-----|")
    for method, stats in report["confidence_stats"].items():
        lines.append(
            f"| {method} | {stats['mean']:.3f} | {stats['std']:.3f} | "
            f"{stats['min']:.3f} | {stats['max']:.3f} |"
        )
    lines.append("")

    # Distribution Comparison
    lines.append("## Category Distribution\n")
    for method, dist in report["distributions"].items():
        lines.append(f"### {method}\n")
        lines.append("| Category | Count | Percentage |")
        lines.append("|----------|-------|------------|")
        for cat, stats in list(dist.items())[:10]:
            lines.append(f"| {cat} | {stats['count']} | {stats['percentage']:.1f}% |")
        lines.append("")

    return "\n".join(lines)


def load_and_compare(
    results_dir: Optional[Path] = None,
    patterns: Optional[List[str]] = None,
) -> Dict:
    """
    Load result files and generate comparison.

    Args:
        results_dir: Directory containing result CSV files
        patterns: List of filename patterns to include (e.g., ["keyword", "llm_openai"])

    Returns:
        Comparison report
    """
    results_dir = Path(results_dir or RESULTS_DIR)

    results_dict = {}

    # Find CSV files
    csv_files = list(results_dir.glob("*.csv"))

    for csv_file in csv_files:
        name = csv_file.stem

        # Filter by patterns if specified
        if patterns:
            if not any(p in name for p in patterns):
                continue

        try:
            results = load_results_from_csv(str(csv_file))
            results_dict[name] = results
            print(f"Loaded {len(results)} results from {csv_file.name}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    if not results_dict:
        print("No result files found!")
        return {}

    return generate_comparison_report(results_dict, results_dir)


def compare_keyword_vs_llm(
    keyword_results: List[ClassificationResult],
    llm_results: List[ClassificationResult],
) -> Dict:
    """
    Specifically compare keyword-based vs LLM-based classification.

    Returns detailed analysis of where they agree and disagree.
    """
    lookup_kw = {r.server_id: r for r in keyword_results}
    lookup_llm = {r.server_id: r for r in llm_results}

    # Servers classified by both
    both_ids = set(lookup_kw.keys()) & set(lookup_llm.keys())

    # Servers only by keyword
    kw_only_ids = set(lookup_kw.keys()) - set(lookup_llm.keys())

    # Servers only by LLM
    llm_only_ids = set(lookup_llm.keys()) - set(lookup_kw.keys())

    # Analyze agreements/disagreements
    agreements = []
    disagreements = []

    for server_id in both_ids:
        kw = lookup_kw[server_id]
        llm = lookup_llm[server_id]

        if kw.category_major == llm.category_major:
            agreements.append({
                "server_id": server_id,
                "category": kw.category_major,
            })
        else:
            disagreements.append({
                "server_id": server_id,
                "server_name": kw.server_name,
                "keyword_category": kw.category_major,
                "keyword_confidence": kw.confidence,
                "keyword_matched": kw.matched_keyword,
                "llm_category": llm.category_major,
                "llm_confidence": llm.confidence,
            })

    return {
        "total_keyword": len(keyword_results),
        "total_llm": len(llm_results),
        "classified_by_both": len(both_ids),
        "keyword_only": len(kw_only_ids),
        "llm_only": len(llm_only_ids),
        "agreement_rate": len(agreements) / len(both_ids) if both_ids else 0,
        "agreements_count": len(agreements),
        "disagreements_count": len(disagreements),
        "disagreements_sample": disagreements[:30],
    }
