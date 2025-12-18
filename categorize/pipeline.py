"""Classification pipeline combining keyword and LLM classifiers."""

from typing import List, Optional, Dict
from pathlib import Path

from .models import MCPServer, ClassificationResult, load_servers_from_json, save_results_to_csv, save_results_to_json
from .classifiers import KeywordClassifier, LLMClassifier
from .config import RESULTS_DIR, INPUT_FILE


class ClassificationPipeline:
    """Pipeline for classifying MCP servers using keyword + LLM fallback."""

    def __init__(
        self,
        llm_provider: str = "openai",
        results_dir: Optional[Path] = None,
    ):
        self.keyword_classifier = KeywordClassifier()
        self.llm_classifier = LLMClassifier(provider=llm_provider)
        self.results_dir = results_dir or RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        servers: List[MCPServer],
        keyword_only: bool = False,
        llm_only: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, List[ClassificationResult]]:
        """
        Run the classification pipeline.

        Args:
            servers: List of MCP servers to classify
            keyword_only: Only use keyword classification
            llm_only: Only use LLM classification (skip keywords)
            show_progress: Show progress bar

        Returns:
            Dictionary with results by method
        """
        results = {
            "keyword": [],
            "llm": [],
            "combined": [],
        }

        if show_progress:
            print(f"Classifying {len(servers)} servers...")

        # Step 1: Keyword classification
        if not llm_only:
            if show_progress:
                print("Step 1: Keyword classification...")

            keyword_results = []
            llm_queue = []

            for server in servers:
                result = self.keyword_classifier.classify(server)
                if result:
                    keyword_results.append(result)
                else:
                    llm_queue.append(server)

            results["keyword"] = keyword_results

            if show_progress:
                print(f"  - Classified by keyword: {len(keyword_results)}/{len(servers)} "
                      f"({len(keyword_results)/len(servers)*100:.1f}%)")

            if keyword_only:
                results["combined"] = keyword_results
                return results

        else:
            llm_queue = servers

        # Step 2: LLM classification for remaining servers
        if llm_queue:
            if show_progress:
                print(f"Step 2: LLM classification for {len(llm_queue)} remaining servers...")

            llm_results = self.llm_classifier.classify_all(llm_queue, show_progress=show_progress)
            results["llm"] = llm_results

            if show_progress:
                print(f"  - Classified by LLM: {len(llm_results)}/{len(llm_queue)}")

        # Step 3: Combine results
        if not llm_only:
            results["combined"] = results["keyword"] + results["llm"]
        else:
            results["combined"] = results["llm"]

        return results

    def run_all_providers(
        self,
        servers: List[MCPServer],
        providers: List[str] = ["openai", "anthropic", "google", "upstage"],
        show_progress: bool = True,
    ) -> Dict[str, List[ClassificationResult]]:
        """
        Run classification with all LLM providers.

        Returns:
            Dictionary with results by provider
        """
        results = {}

        # First, run keyword classification
        if show_progress:
            print("Running keyword classification...")

        keyword_results = []
        for server in servers:
            result = self.keyword_classifier.classify(server)
            if result:
                keyword_results.append(result)

        results["keyword"] = keyword_results
        if show_progress:
            print(f"  - Keyword: {len(keyword_results)}/{len(servers)}")

        # Run each LLM provider on ALL servers (for comparison)
        for provider in providers:
            try:
                if show_progress:
                    print(f"\nRunning {provider} classification...")

                llm_clf = LLMClassifier(provider=provider)
                llm_results = llm_clf.classify_all(servers, show_progress=show_progress)
                results[f"llm_{provider}"] = llm_results

                if show_progress:
                    print(f"  - {provider}: {len(llm_results)}/{len(servers)}")

            except Exception as e:
                print(f"Error with {provider}: {e}")
                results[f"llm_{provider}"] = []

        return results

    def save_results(
        self,
        results: Dict[str, List[ClassificationResult]],
        prefix: str = "",
    ) -> Dict[str, Path]:
        """
        Save classification results to files.

        Returns:
            Dictionary mapping result type to file path
        """
        saved_files = {}

        for result_type, result_list in results.items():
            if not result_list:
                continue

            # Generate filename
            filename = f"{prefix}_{result_type}" if prefix else result_type

            # Save CSV
            csv_path = self.results_dir / f"{filename}.csv"
            save_results_to_csv(result_list, str(csv_path))
            saved_files[f"{result_type}_csv"] = csv_path

            # Save JSON
            json_path = self.results_dir / f"{filename}.json"
            save_results_to_json(result_list, str(json_path))
            saved_files[f"{result_type}_json"] = json_path

        return saved_files

    def get_statistics(
        self, results: List[ClassificationResult]
    ) -> Dict:
        """Get statistics about classification results."""
        if not results:
            return {}

        by_major = {}
        by_minor = {}
        by_method = {}
        confidences = []

        for r in results:
            by_major[r.category_major] = by_major.get(r.category_major, 0) + 1

            key = (r.category_major, r.category_minor)
            by_minor[key] = by_minor.get(key, 0) + 1

            method = r.method.split("_")[0]  # "keyword" or "llm"
            by_method[method] = by_method.get(method, 0) + 1

            confidences.append(r.confidence)

        return {
            "total": len(results),
            "by_major_category": dict(sorted(by_major.items(), key=lambda x: -x[1])),
            "by_minor_category": {f"{k[0]}/{k[1]}": v for k, v in sorted(by_minor.items(), key=lambda x: -x[1])},
            "by_method": by_method,
            "confidence": {
                "mean": sum(confidences) / len(confidences),
                "min": min(confidences),
                "max": max(confidences),
            }
        }


def run_pipeline(
    input_file: Optional[str] = None,
    provider: str = "openai",
    keyword_only: bool = False,
    llm_only: bool = False,
    all_providers: bool = False,
    save: bool = True,
    show_progress: bool = True,
) -> Dict[str, List[ClassificationResult]]:
    """
    Convenience function to run the full pipeline.

    Args:
        input_file: Path to input JSON file (default: config.INPUT_FILE)
        provider: LLM provider to use
        keyword_only: Only use keyword classification
        llm_only: Only use LLM classification
        all_providers: Run all LLM providers for comparison
        save: Save results to files
        show_progress: Show progress

    Returns:
        Dictionary with classification results
    """
    # Load servers
    filepath = input_file or str(INPUT_FILE)
    if show_progress:
        print(f"Loading servers from {filepath}...")

    servers = load_servers_from_json(filepath)
    if show_progress:
        print(f"Loaded {len(servers)} servers")

    # Create pipeline
    pipeline = ClassificationPipeline(llm_provider=provider)

    # Run classification
    if all_providers:
        results = pipeline.run_all_providers(servers, show_progress=show_progress)
    else:
        results = pipeline.run(
            servers,
            keyword_only=keyword_only,
            llm_only=llm_only,
            show_progress=show_progress,
        )

    # Save results
    if save:
        if show_progress:
            print("\nSaving results...")

        saved_files = pipeline.save_results(results)
        for name, path in saved_files.items():
            if show_progress:
                print(f"  - {name}: {path}")

    # Print statistics
    if show_progress and "combined" in results:
        print("\nStatistics:")
        stats = pipeline.get_statistics(results["combined"])
        print(f"  Total classified: {stats.get('total', 0)}")
        print(f"  By method: {stats.get('by_method', {})}")
        print(f"  Top categories: {list(stats.get('by_major_category', {}).items())[:5]}")

    return results
