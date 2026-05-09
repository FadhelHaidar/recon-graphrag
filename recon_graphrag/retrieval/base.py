"""Abstract base class for retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from recon_graphrag.models.types import SearchResult


class BaseRetriever(ABC):
    """Abstract retriever interface.

    All search modes (local, global, drift) implement this interface.
    """

    @abstractmethod
    async def search(self, query: str, **kwargs) -> SearchResult:
        """Run a search and return structured results."""
        ...
