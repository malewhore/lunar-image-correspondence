"""Fine-tuned lunar LightGlue placeholder — do not implement until baseline is frozen."""

from backend.matchers.base import MatcherInterface
from backend.schemas.matching import MatchingResult
from backend.schemas.processed_pair import ProcessedPair


class LightGlueLunarMatcher(MatcherInterface):
    def match(self, pair: ProcessedPair) -> MatchingResult:
        raise NotImplementedError(
            "Fine-tuned lunar LightGlue comes after CHECKPOINT: baseline frozen. "
            "Use LightGlueBaselineMatcher first."
        )
