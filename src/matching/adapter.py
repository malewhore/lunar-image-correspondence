import time
import numpy as np
import torch
import cv2
from typing import Optional

from src.matching.matcher import DeepMatcher
from src.schemas.processed_pair import PreprocessedPair
from src.schemas.algorithm_result import AlgorithmResult
from src.evaluation.metrics import BaselineMetrics

class AlgorithmAdapter:
    """
    Wraps the core matching logic to ingest a PreprocessedPair and return 
    a standardized AlgorithmResult for the Registration Orchestrator.
    """
    def __init__(self, matcher: DeepMatcher):
        self.matcher = matcher
        
    def process(self, pair: PreprocessedPair) -> AlgorithmResult:
        start_time = time.time()
        
        from src.ingestion.processed_pair_adapter import PreprocessedPairAdapter
        pp_adapter = PreprocessedPairAdapter()
        img0_tensor, img1_tensor = pp_adapter.adapt(pair)
        
        if self.matcher.device:
            img0_tensor = img0_tensor.to(self.matcher.device)
            img1_tensor = img1_tensor.to(self.matcher.device)
            
        results = self.matcher.extract_and_match(img0_tensor, img1_tensor)
        
        m_kpts0 = results['m_kpts0'].cpu().numpy()
        m_kpts1 = results['m_kpts1'].cpu().numpy()
        
        inference_time = time.time() - start_time
        
        # We need the transformation matrix and inliers mask. 
        # BaselineMetrics.compute_metrics uses findFundamentalMat to get inliers, 
        # but the orchestrator might need Homography. We will calculate both if needed, 
        # or use findHomography directly.
        num_matches = len(m_kpts0)
        inlier_mask = np.zeros(num_matches, dtype=bool)
        H = None
        metrics = {
            'candidate_matches': num_matches,
            'inference_time': inference_time,
            'inliers': 0,
            'inlier_ratio': 0.0
        }
        
        if num_matches >= 4:
            # Calculate Homography using RANSAC for actual registration
            H, mask = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, 3.0)
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                inliers = int(np.sum(inlier_mask))
                metrics['inliers'] = inliers
                metrics['inlier_ratio'] = inliers / num_matches if num_matches > 0 else 0.0
        
        return AlgorithmResult(
            pair_id=pair.pair_id,
            algorithm_name="SuperPoint+LightGlue",
            matched_kpts_moving=m_kpts0,
            matched_kpts_reference=m_kpts1,
            inlier_mask=inlier_mask,
            transformation_matrix=H,
            metrics=metrics
        )
