"""Base classifier interface."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models import MCPServer, ClassificationResult


class BaseClassifier(ABC):
    """Abstract base class for classifiers."""

    @abstractmethod
    def classify(self, server: MCPServer) -> Optional[ClassificationResult]:
        """
        Classify a single MCP server.

        Args:
            server: The MCP server to classify

        Returns:
            ClassificationResult if classification successful, None otherwise
        """
        pass

    def classify_batch(self, servers: List[MCPServer]) -> List[ClassificationResult]:
        """
        Classify multiple MCP servers.

        Args:
            servers: List of MCP servers to classify

        Returns:
            List of ClassificationResults
        """
        results = []
        for server in servers:
            result = self.classify(server)
            if result:
                results.append(result)
        return results

    @property
    @abstractmethod
    def name(self) -> str:
        """Get classifier name."""
        pass
