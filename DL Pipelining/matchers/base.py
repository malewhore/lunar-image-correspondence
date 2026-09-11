from abc import ABC, abstractmethod

from backend.schemas.matching import MatchingResult
from backend.schemas.processed_pair import ProcessedPair


class MatcherInterface(ABC):
    @abstractmethod
    def match(self, pair: ProcessedPair) -> MatchingResult:
        """Run correspondence on a verified ProcessedPair."""
