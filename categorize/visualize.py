"""Visualization functions for MCP Tool Classification results."""

from typing import List, Dict, Optional
from pathlib import Path
import json

from .models import ClassificationResult, load_results_from_csv
from .config import RESULTS_DIR
from .taxonomy import CATEGORY_TAXONOMY


def plot_major_category_distribution(
    results: List[ClassificationResult],
    title: str = "MCP Server Distribution by Major Category",
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8),
) -> None:
    """
    Create a pie chart of major category distribution.

    Args:
        results: List of classification results
        title: Chart title
        output_path: Path to save the figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    from collections import Counter

    # Count categories
    categories = [r.category_major for r in results]
    counter = Counter(categories)

    # Sort by count
    sorted_items = counter.most_common()
    labels = [item[0] for item in sorted_items]
    sizes = [item[1] for item in sorted_items]

    # Create pie chart
    fig, ax = plt.subplots(figsize=figsize)

    # Colors
    colors = plt.cm.tab20(range(len(labels)))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes))})',
        colors=colors,
        startangle=90,
        pctdistance=0.75,
    )

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()


def plot_subcategory_bars(
    results: List[ClassificationResult],
    output_path: Optional[str] = None,
    figsize: tuple = (16, 20),
    top_n: int = 10,
) -> None:
    """
    Create bar charts of subcategory distribution for each major category.

    Args:
        results: List of classification results
        output_path: Path to save the figure
        figsize: Figure size
        top_n: Number of top subcategories to show per major category
    """
    import matplotlib.pyplot as plt
    from collections import Counter

    # Group by major category
    major_categories = list(CATEGORY_TAXONOMY.keys())
    n_major = len(major_categories)

    # Create subplots
    n_cols = 2
    n_rows = (n_major + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, major in enumerate(major_categories):
        ax = axes[idx]

        # Filter results for this major category
        major_results = [r for r in results if r.category_major == major]

        if not major_results:
            ax.set_title(f"{major} (0)")
            ax.set_visible(False)
            continue

        # Count subcategories
        subcats = [r.category_minor for r in major_results]
        counter = Counter(subcats)
        top_items = counter.most_common(top_n)

        if not top_items:
            ax.set_visible(False)
            continue

        labels = [item[0] for item in top_items]
        values = [item[1] for item in top_items]

        # Create horizontal bar chart
        bars = ax.barh(labels, values, color=plt.cm.viridis(idx / n_major))
        ax.set_title(f"{major} ({len(major_results)})", fontweight='bold')
        ax.set_xlabel('Count')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   str(val), va='center', fontsize=8)

    # Hide unused subplots
    for idx in range(n_major, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('MCP Server Distribution by Subcategory', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()


def plot_tools_per_category(
    results: List[ClassificationResult],
    output_path: Optional[str] = None,
    figsize: tuple = (14, 6),
) -> None:
    """
    Create bar chart showing average tools per server by category.

    Args:
        results: List of classification results
        output_path: Path to save the figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # Calculate average tools per category
    tools_by_category = defaultdict(list)

    for r in results:
        tools_by_category[r.category_major].append(r.tools_count)

    # Calculate averages
    categories = []
    averages = []
    counts = []

    for cat in sorted(tools_by_category.keys()):
        tools = tools_by_category[cat]
        categories.append(cat)
        averages.append(sum(tools) / len(tools))
        counts.append(len(tools))

    # Create bar chart
    fig, ax = plt.subplots(figsize=figsize)

    x = range(len(categories))
    bars = ax.bar(x, averages, color=plt.cm.viridis(range(len(categories))))

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Average Tools per Server')
    ax.set_title('Average Number of Tools per Server by Category', fontweight='bold')

    # Add value labels
    for bar, avg, count in zip(bars, averages, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{avg:.1f}\n(n={count})', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()


def plot_model_comparison(
    results_dict: Dict[str, List[ClassificationResult]],
    output_path: Optional[str] = None,
    figsize: tuple = (16, 8),
) -> None:
    """
    Create grouped bar chart comparing category distributions across models.

    Args:
        results_dict: Dictionary mapping model name to results
        output_path: Path to save the figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    # Get all categories
    all_categories = set()
    for results in results_dict.values():
        for r in results:
            all_categories.add(r.category_major)

    categories = sorted(all_categories)
    models = list(results_dict.keys())

    # Create data matrix
    data = []
    for model in models:
        results = results_dict[model]
        counter = Counter(r.category_major for r in results)
        total = len(results)
        percentages = [counter.get(cat, 0) / total * 100 if total > 0 else 0
                      for cat in categories]
        data.append(percentages)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(categories))
    width = 0.8 / len(models)

    for i, (model, percentages) in enumerate(zip(models, data)):
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, percentages, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Category Distribution Comparison Across Models', fontweight='bold')
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()


def plot_confidence_distribution(
    results_dict: Dict[str, List[ClassificationResult]],
    output_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> None:
    """
    Create box plot of confidence distributions by model.

    Args:
        results_dict: Dictionary mapping model name to results
        output_path: Path to save the figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    # Prepare data
    data = []
    labels = []

    for model, results in results_dict.items():
        if results:
            confidences = [r.confidence for r in results]
            data.append(confidences)
            labels.append(model)

    # Create box plot
    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    # Color boxes
    colors = plt.cm.Set3(range(len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Confidence Score')
    ax.set_title('Confidence Score Distribution by Classification Method', fontweight='bold')

    # Add mean markers
    for i, d in enumerate(data):
        mean = sum(d) / len(d)
        ax.scatter(i + 1, mean, marker='D', color='red', s=50, zorder=5, label='Mean' if i == 0 else '')

    ax.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()


def plot_agreement_heatmap(
    agreement_matrix: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> None:
    """
    Create heatmap of pairwise agreement rates.

    Args:
        agreement_matrix: Nested dict of agreement rates
        output_path: Path to save the figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    import numpy as np

    models = list(agreement_matrix.keys())
    n = len(models)

    # Create matrix
    matrix = np.zeros((n, n))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                matrix[i, j] = 1.0
            else:
                key = f"{m1}_vs_{m2}" if f"{m1}_vs_{m2}" in agreement_matrix else f"{m2}_vs_{m1}"
                matrix[i, j] = agreement_matrix.get(key, {}).get("agreement_rate", 0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Agreement Rate', rotation=-90, va="bottom")

    # Set ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_yticklabels(models)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black")

    ax.set_title('Pairwise Agreement Rate Heatmap', fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()


def generate_all_visualizations(
    results_dict: Dict[str, List[ClassificationResult]],
    output_dir: Optional[Path] = None,
) -> List[str]:
    """
    Generate all visualization charts.

    Args:
        results_dict: Dictionary mapping method name to results
        output_dir: Directory to save figures

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir or RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Use combined results if available, otherwise use first available
    main_results = results_dict.get("combined", list(results_dict.values())[0])

    # 1. Major category pie chart
    pie_path = output_dir / "major_category_pie.png"
    plot_major_category_distribution(main_results, output_path=str(pie_path))
    saved_files.append(str(pie_path))

    # 2. Subcategory bar charts
    subcat_path = output_dir / "subcategory_bars.png"
    plot_subcategory_bars(main_results, output_path=str(subcat_path))
    saved_files.append(str(subcat_path))

    # 3. Tools per category
    tools_path = output_dir / "tools_per_category.png"
    plot_tools_per_category(main_results, output_path=str(tools_path))
    saved_files.append(str(tools_path))

    # 4. Model comparison (if multiple methods)
    if len(results_dict) > 1:
        compare_path = output_dir / "model_comparison.png"
        plot_model_comparison(results_dict, output_path=str(compare_path))
        saved_files.append(str(compare_path))

        # 5. Confidence distribution
        conf_path = output_dir / "confidence_distribution.png"
        plot_confidence_distribution(results_dict, output_path=str(conf_path))
        saved_files.append(str(conf_path))

    return saved_files


def visualize_from_files(
    results_dir: Optional[Path] = None,
    patterns: Optional[List[str]] = None,
) -> List[str]:
    """
    Load result files and generate visualizations.

    Args:
        results_dir: Directory containing result CSV files
        patterns: List of filename patterns to include

    Returns:
        List of saved file paths
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
        return []

    return generate_all_visualizations(results_dict, results_dir)
