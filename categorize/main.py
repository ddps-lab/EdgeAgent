#!/usr/bin/env python3
"""
MCP Tool Categorization - Main CLI

Usage:
    python main.py                      # Run with default settings (keyword + OpenAI)
    python main.py --keyword-only       # Only run keyword classification
    python main.py --provider anthropic # Use Claude instead of GPT-4o
    python main.py --all-providers      # Run all LLM providers for comparison
    python main.py --compare            # Compare existing results
    python main.py --visualize          # Generate visualizations from results
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from categorize.config import INPUT_FILE, RESULTS_DIR, DEFAULT_PROVIDER
from categorize.models import load_servers_from_json
from categorize.pipeline import ClassificationPipeline, run_pipeline
from categorize.compare import load_and_compare, generate_comparison_report
from categorize.visualize import visualize_from_files, generate_all_visualizations


def main():
    parser = argparse.ArgumentParser(
        description="MCP Tool Categorization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                        # Default: keyword + OpenAI fallback
    python main.py --keyword-only         # Only keyword classification
    python main.py --llm-only             # Only LLM classification (skip keywords)
    python main.py --provider anthropic   # Use Claude for LLM
    python main.py --all-providers        # Compare all LLM providers
    python main.py --compare              # Compare existing result files
    python main.py --visualize            # Generate charts from results
        """
    )

    # Input/Output options
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=str(INPUT_FILE),
        help=f"Input JSON file path (default: {INPUT_FILE})"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help=f"Output directory for results (default: {RESULTS_DIR})"
    )

    # Classification options
    parser.add_argument(
        "--keyword-only",
        action="store_true",
        help="Only use keyword classification (no LLM)"
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Only use LLM classification (skip keywords)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "google", "upstage"],
        default=DEFAULT_PROVIDER,
        help=f"LLM provider to use (default: {DEFAULT_PROVIDER})"
    )
    parser.add_argument(
        "--all-providers",
        action="store_true",
        help="Run classification with all LLM providers for comparison"
    )

    # Analysis options
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare existing result files"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations from existing results"
    )

    # Other options
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show keyword classification statistics (no actual classification)"
    )

    args = parser.parse_args()

    # Set output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mode: Compare existing results
    if args.compare:
        print("Comparing existing results...")
        report = load_and_compare(output_dir)
        if report:
            print("\nComparison complete!")
        return

    # Mode: Generate visualizations
    if args.visualize:
        print("Generating visualizations...")
        saved = visualize_from_files(output_dir)
        if saved:
            print(f"\nGenerated {len(saved)} visualization(s)")
        return

    # Mode: Stats only
    if args.stats_only:
        print(f"Loading servers from {args.input}...")
        servers = load_servers_from_json(args.input)
        print(f"Loaded {len(servers)} servers")

        from categorize.classifiers import KeywordClassifier
        classifier = KeywordClassifier()
        stats = classifier.get_coverage_stats(servers)

        print("\nKeyword Classification Coverage Statistics:")
        print(f"  Total servers: {stats['total']}")
        print(f"  Classified: {stats['classified']} ({stats['percentage']}%)")
        print(f"  By method:")
        for method, count in stats['by_method'].items():
            print(f"    - {method}: {count}")
        print(f"  By category (top 10):")
        sorted_cats = sorted(stats['by_category'].items(), key=lambda x: -x[1])
        for cat, count in sorted_cats[:10]:
            print(f"    - {cat}: {count}")
        return

    # Mode: Full classification
    print("=" * 60)
    print("MCP Tool Categorization System")
    print("=" * 60)

    # Run pipeline
    if args.all_providers:
        providers = ["openai", "anthropic", "google", "upstage"]
        print(f"\nRunning classification with all providers: {providers}")

        # Load servers once
        print(f"\nLoading servers from {args.input}...")
        servers = load_servers_from_json(args.input)
        print(f"Loaded {len(servers)} servers")

        # Create pipeline and run
        pipeline = ClassificationPipeline(results_dir=output_dir)
        results = pipeline.run_all_providers(
            servers,
            providers=providers,
            show_progress=not args.quiet,
        )

        # Save results
        if not args.no_save:
            print("\nSaving results...")
            saved_files = pipeline.save_results(results)
            for name, path in saved_files.items():
                print(f"  - {name}: {path}")

            # Generate comparison report
            print("\nGenerating comparison report...")
            generate_comparison_report(results, output_dir)

            # Generate visualizations
            print("\nGenerating visualizations...")
            generate_all_visualizations(results, output_dir)

    else:
        # Single provider run
        results = run_pipeline(
            input_file=args.input,
            provider=args.provider,
            keyword_only=args.keyword_only,
            llm_only=args.llm_only,
            save=not args.no_save,
            show_progress=not args.quiet,
        )

        # Generate visualizations if we saved results
        if not args.no_save and results.get("combined"):
            print("\nGenerating visualizations...")
            from categorize.visualize import generate_all_visualizations
            saved = generate_all_visualizations(results, output_dir)
            print(f"Generated {len(saved)} visualization(s)")

    print("\n" + "=" * 60)
    print("Classification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
