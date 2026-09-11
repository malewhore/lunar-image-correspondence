import time
import numpy as np
import cv2

class BaselineMetrics:
    @staticmethod
    def compute_metrics(matches, kpts0, kpts1, inference_time):
        """
        Compute basic baseline metrics including RANSAC inlier verification.
        """
        num_matches = len(matches)
        
        inliers = 0
        inlier_ratio = 0.0
        
        if num_matches >= 4:
            src_pts = np.float32(kpts0).reshape(-1, 1, 2)
            dst_pts = np.float32(kpts1).reshape(-1, 1, 2)
            
            # Find fundamental matrix using RANSAC to get inliers
            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 3.0, 0.99)
            if mask is not None:
                inliers = np.sum(mask)
                inlier_ratio = inliers / num_matches if num_matches > 0 else 0.0
                
        return {
            'candidate_matches': num_matches,
            'inliers': int(inliers),
            'inlier_ratio': float(inlier_ratio),
            'inference_time': inference_time
        }
