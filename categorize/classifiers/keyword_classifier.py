"""Keyword-based classifier for MCP servers."""

from typing import Optional, Tuple, List
import re

from .base import BaseClassifier
from ..models import MCPServer, ClassificationResult
from ..taxonomy import KEYWORD_RULES, CATEGORY_TAXONOMY


class KeywordClassifier(BaseClassifier):
    """Classifier that uses keyword rules to categorize MCP servers."""

    def __init__(self):
        self.rules = KEYWORD_RULES

    @property
    def name(self) -> str:
        return "keyword"

    def classify(self, server: MCPServer) -> Optional[ClassificationResult]:
        """
        Classify a server using keyword matching.

        Priority:
        1. Exact match on server_id or server_name
        2. Contains match on server_id or server_name
        3. Description keywords match on tools descriptions
        """
        server_id_lower = server.server_id.lower()
        server_name_lower = server.server_name.lower()
        combined_text = f"{server_id_lower} {server_name_lower}"

        # Try classification in order of priority
        result = self._try_exact_match(server, server_id_lower, server_name_lower)
        if result:
            return result

        result = self._try_contains_match(server, combined_text)
        if result:
            return result

        result = self._try_description_match(server)
        if result:
            return result

        return None

    def _try_exact_match(
        self, server: MCPServer, server_id_lower: str, server_name_lower: str
    ) -> Optional[ClassificationResult]:
        """Try exact match on server_id or server_name."""
        for (major, minor), rules in self.rules.items():
            exact_matches = rules.get("exact_match", [])
            for keyword in exact_matches:
                keyword_lower = keyword.lower()
                if keyword_lower == server_id_lower or keyword_lower == server_name_lower:
                    return self._create_result(
                        server, major, minor, 1.0, f"exact:{keyword}"
                    )
                # Also check if the keyword is contained as a whole word
                if re.search(rf'\b{re.escape(keyword_lower)}\b', server_id_lower) or \
                   re.search(rf'\b{re.escape(keyword_lower)}\b', server_name_lower):
                    return self._create_result(
                        server, major, minor, 0.95, f"exact:{keyword}"
                    )
        return None

    def _try_contains_match(
        self, server: MCPServer, combined_text: str
    ) -> Optional[ClassificationResult]:
        """Try contains match on combined server_id and server_name."""
        best_match: Optional[Tuple[str, str, float, str]] = None

        for (major, minor), rules in self.rules.items():
            contains_keywords = rules.get("contains", [])
            exclude_keywords = rules.get("exclude", [])

            # Check exclusions first
            should_exclude = False
            for exclude in exclude_keywords:
                if exclude.lower() in combined_text:
                    should_exclude = True
                    break

            if should_exclude:
                continue

            for keyword in contains_keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in combined_text:
                    # Longer keyword matches are more specific
                    confidence = 0.9 - (0.01 * (20 - len(keyword)))
                    confidence = max(0.7, min(0.9, confidence))

                    if best_match is None or confidence > best_match[2]:
                        best_match = (major, minor, confidence, f"contains:{keyword}")

        if best_match:
            return self._create_result(
                server, best_match[0], best_match[1], best_match[2], best_match[3]
            )
        return None

    def _try_description_match(
        self, server: MCPServer
    ) -> Optional[ClassificationResult]:
        """Try matching based on tool descriptions."""
        all_descriptions = server.get_all_descriptions()
        if not all_descriptions:
            return None

        # Count keyword matches for each category
        category_scores: dict = {}

        for (major, minor), rules in self.rules.items():
            desc_keywords = rules.get("description_keywords", [])
            exclude_keywords = rules.get("exclude", [])

            # Check exclusions
            should_exclude = False
            for exclude in exclude_keywords:
                if exclude.lower() in all_descriptions:
                    should_exclude = True
                    break

            if should_exclude:
                continue

            score = 0
            matched_keywords = []
            for keyword in desc_keywords:
                keyword_lower = keyword.lower()
                # Count occurrences
                count = all_descriptions.count(keyword_lower)
                if count > 0:
                    score += count
                    matched_keywords.append(keyword)

            if score > 0:
                category_scores[(major, minor)] = (score, matched_keywords)

        if not category_scores:
            return None

        # Get the category with highest score
        best_category = max(category_scores.items(), key=lambda x: x[1][0])
        (major, minor), (score, keywords) = best_category

        # Confidence based on score (more matches = higher confidence)
        confidence = min(0.8, 0.5 + (score * 0.05))

        return self._create_result(
            server, major, minor, confidence, f"desc:{','.join(keywords[:3])}"
        )

    def _create_result(
        self,
        server: MCPServer,
        major: str,
        minor: str,
        confidence: float,
        matched_keyword: str,
    ) -> ClassificationResult:
        """Create a classification result."""
        return ClassificationResult(
            server_id=server.server_id,
            server_name=server.server_name,
            category_major=major,
            category_minor=minor,
            tools_count=server.tools_count,
            confidence=confidence,
            method="keyword",
            matched_keyword=matched_keyword,
            source=server.source,
        )

    def get_coverage_stats(self, servers: List[MCPServer]) -> dict:
        """Get statistics about keyword classification coverage."""
        total = len(servers)
        classified = 0
        by_method = {"exact": 0, "contains": 0, "desc": 0}
        by_category = {}

        for server in servers:
            result = self.classify(server)
            if result:
                classified += 1

                # Track method
                if result.matched_keyword:
                    if result.matched_keyword.startswith("exact:"):
                        by_method["exact"] += 1
                    elif result.matched_keyword.startswith("contains:"):
                        by_method["contains"] += 1
                    elif result.matched_keyword.startswith("desc:"):
                        by_method["desc"] += 1

                # Track category
                cat_key = result.category_major
                by_category[cat_key] = by_category.get(cat_key, 0) + 1

        return {
            "total": total,
            "classified": classified,
            "percentage": round(classified / total * 100, 2) if total > 0 else 0,
            "by_method": by_method,
            "by_category": by_category,
        }
