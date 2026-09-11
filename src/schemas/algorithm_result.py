from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class AlgorithmResult:
    """
    The standardized output contract required by the Registration Orchestrator.
    This guarantees that the backend/frontend can safely consume the matching output 
    regardless of whether SIFT, SuperPoint, or another algorithm was used.
    """
    pair_id: str
    algorithm_name: str
    
    # N x 2 arrays containing the matched keypoints in the original image coordinate space
    matched_kpts_moving: np.ndarray
    matched_kpts_reference: np.ndarray
    
    # 1D boolean array indicating which matches survived geometric verification
    inlier_mask: np.ndarray
    
    # Estimated transformation matrix (e.g. 3x3 Homography or 2x3 Affine)
    transformation_matrix: Optional[np.ndarray]
    
    # Key evaluation metrics (inliers, precision approximations, runtime)
    metrics: Dict[str, Any]
    
    def get_inliers_moving(self) -> np.ndarray:
        return self.matched_kpts_moving[self.inlier_mask]
        
    def get_inliers_reference(self) -> np.ndarray:
        return self.matched_kpts_reference[self.inlier_mask]
