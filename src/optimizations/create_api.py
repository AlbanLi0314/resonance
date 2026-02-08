"""
Create Production API
=====================
Generates a production-ready API wrapper for the academic matcher.
"""

from pathlib import Path


def create_api_module(output_file: Path = None) -> dict:
    """
    Create a production API module.

    Args:
        output_file: Path to save the API module

    Returns:
        Result dict
    """

    api_code = '''"""
Academic Matcher API
====================
Production-ready API for academic researcher matching.

Usage:
    from src.api import AcademicMatcherAPI

    api = AcademicMatcherAPI()
    api.initialize()
    results = api.search("DNA origami self-assembly")
"""

import json
from typing import List, Dict, Optional
from pathlib import Path

from .matcher import AcademicMatcher
from .config import GEMINI_API_KEY


class AcademicMatcherAPI:
    """
    Production API wrapper for academic matcher.

    Features:
    - Simple initialization
    - Search with configurable options
    - Batch search
    - Health check
    - Metadata retrieval
    """

    def __init__(self):
        self.matcher: Optional[AcademicMatcher] = None
        self.initialized = False
        self.config = {}

    def initialize(self, skip_rerank: bool = False) -> bool:
        """
        Initialize the API.

        Args:
            skip_rerank: If True, disable LLM reranking

        Returns:
            True if initialization succeeded
        """
        try:
            self.matcher = AcademicMatcher()
            self.matcher.initialize(skip_rerank=skip_rerank)
            self.initialized = True
            self.config = {
                "reranking_enabled": not skip_rerank,
                "num_researchers": len(self.matcher.researchers),
            }
            return True
        except Exception as e:
            print(f"Initialization failed: {e}")
            return False

    def health_check(self) -> Dict:
        """
        Check API health.

        Returns:
            Health status dict
        """
        return {
            "status": "healthy" if self.initialized else "not_initialized",
            "initialized": self.initialized,
            "config": self.config
        }

    def search(self, query: str, top_k: int = 5, skip_rerank: bool = None) -> Dict:
        """
        Search for researchers.

        Args:
            query: Search query string
            top_k: Number of results to return
            skip_rerank: Override reranking setting

        Returns:
            Search results dict
        """
        if not self.initialized:
            return {"error": "API not initialized", "results": []}

        try:
            if skip_rerank is None:
                skip_rerank = not self.config.get("reranking_enabled", False)

            results = self.matcher.search(query, final_k=top_k, skip_rerank=skip_rerank)

            return {
                "query": query,
                "num_results": len(results),
                "results": results,
                "reranked": not skip_rerank
            }
        except Exception as e:
            return {"error": str(e), "results": []}

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[Dict]:
        """
        Search for multiple queries.

        Args:
            queries: List of query strings
            top_k: Number of results per query

        Returns:
            List of search result dicts
        """
        return [self.search(q, top_k) for q in queries]

    def get_researcher(self, researcher_id: str) -> Optional[Dict]:
        """
        Get researcher by ID.

        Args:
            researcher_id: Researcher ID

        Returns:
            Researcher dict or None
        """
        if not self.initialized:
            return None

        for r in self.matcher.researchers:
            if r.get("id") == researcher_id:
                return {
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "position": r.get("position"),
                    "department": r.get("department"),
                    "lab": r.get("lab"),
                    "email": r.get("email"),
                    "personal_website": r.get("personal_website"),
                    "research_interests": r.get("research_interests"),
                }
        return None

    def list_researchers(self, limit: int = 100, offset: int = 0) -> Dict:
        """
        List researchers.

        Args:
            limit: Max researchers to return
            offset: Starting offset

        Returns:
            List of researcher summaries
        """
        if not self.initialized:
            return {"error": "API not initialized", "researchers": []}

        researchers = self.matcher.researchers[offset:offset+limit]

        return {
            "total": len(self.matcher.researchers),
            "limit": limit,
            "offset": offset,
            "researchers": [
                {
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "department": r.get("department"),
                    "position": r.get("position"),
                }
                for r in researchers
            ]
        }

    def get_stats(self) -> Dict:
        """
        Get API statistics.

        Returns:
            Statistics dict
        """
        if not self.initialized:
            return {"error": "API not initialized"}

        # Count by department
        departments = {}
        positions = {}

        for r in self.matcher.researchers:
            dept = r.get("department", "Unknown")
            pos = r.get("position", "Unknown")
            departments[dept] = departments.get(dept, 0) + 1
            positions[pos] = positions.get(pos, 0) + 1

        return {
            "total_researchers": len(self.matcher.researchers),
            "by_department": departments,
            "by_position": positions,
        }


# Singleton instance for convenience
_api_instance = None


def get_api(skip_rerank: bool = False) -> AcademicMatcherAPI:
    """Get or create API instance."""
    global _api_instance
    if _api_instance is None:
        _api_instance = AcademicMatcherAPI()
        _api_instance.initialize(skip_rerank=skip_rerank)
    return _api_instance


# Quick test
if __name__ == "__main__":
    api = AcademicMatcherAPI()
    print("Initializing API...")

    if api.initialize(skip_rerank=True):
        print("API initialized successfully!")
        print(f"Health: {api.health_check()}")
        print(f"Stats: {api.get_stats()}")

        result = api.search("DNA origami self-assembly", top_k=3)
        print(f"Search results: {json.dumps(result, indent=2)}")
    else:
        print("Failed to initialize API")
'''

    output_file = output_file or Path(__file__).parent.parent / "api.py"
    output_file = Path(output_file)

    with open(output_file, 'w') as f:
        f.write(api_code)

    print(f"API module created: {output_file}")

    return {
        "output_file": str(output_file),
    }


if __name__ == "__main__":
    result = create_api_module()
    print(f"Result: {result}")
