from dataclasses import dataclass
import numpy as np
from typing import Dict, Any

@dataclass
class PreprocessedPair:
    """
    The input contract provided by the CV team based on their preprocessing module.
    """
    pair_id: str
    representation: str
    moving: np.ndarray
    reference: np.ndarray
    metadata: Dict[str, Any]
