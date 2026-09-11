import torch
import torch.nn as nn

class GeometricMatchLoss(nn.Module):
    def __init__(self, match_threshold=3.0):
        super().__init__()
        self.match_threshold = match_threshold
        self.bce = nn.BCEWithLogitsLoss()

    def warp_keypoints(self, kpts, H):
        """
        Warps keypoints using the homography H.
        kpts: [N, 2]
        H: [3, 3]
        """
        if len(kpts) == 0:
            return kpts
        
        N = kpts.shape[0]
        # Make homogeneous coordinates [N, 3]
        kpts_h = torch.cat([kpts, torch.ones(N, 1, device=kpts.device)], dim=1)
        
        # Warp
        warped_h = (H @ kpts_h.T).T  # [N, 3]
        
        # Normalize
        warped = warped_h[:, :2] / (warped_h[:, 2:] + 1e-8)
        return warped

    def forward(self, kpts0, kpts1, matching_scores0, H_gt):
        """
        kpts0: [N, 2]
        kpts1: [M, 2]
        matching_scores0: [N] probability of each kpt0 having a match
        H_gt: [3, 3]
        """
        N = kpts0.shape[0]
        M = kpts1.shape[0]
        
        if N == 0 or M == 0:
            return torch.tensor(0.0, device=kpts0.device, requires_grad=True)

        # 1. Warp kpts0
        kpts0_warped = self.warp_keypoints(kpts0, H_gt)
        
        # 2. Pairwise distances
        dist = torch.norm(kpts0_warped.unsqueeze(1) - kpts1.unsqueeze(0), dim=-1) # [N, M]
        
        # 3. Ground truth match exists if minimum distance < threshold
        min_dist, _ = torch.min(dist, dim=1) # [N]
        gt_match_exists = (min_dist < self.match_threshold).float()
        
        # 4. BCE Loss
        # matching_scores0 are already probabilities or logits? LightGlue scores are usually [0, 1]
        loss = nn.functional.binary_cross_entropy(matching_scores0, gt_match_exists)
        
        return loss
