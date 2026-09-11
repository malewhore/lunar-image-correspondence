import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

class DeepMatcher:
    def __init__(self, max_keypoints=2048, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

    def extract_and_match(self, image0: torch.Tensor, image1: torch.Tensor):
        """
        Extract keypoints and match two images.
        image0, image1: [1, C, H, W] tensors
        """
        # Extract features
        feats0 = self.extractor.extract(image0.to(self.device))
        feats1 = self.extractor.extract(image1.to(self.device))

        # Match
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        
        # Remove batch dimension
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

        kpts0 = feats0['keypoints']
        kpts1 = feats1['keypoints']
        matches = matches01['matches']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return {
            'kpts0': kpts0,
            'kpts1': kpts1,
            'matches': matches,
            'm_kpts0': m_kpts0,
            'm_kpts1': m_kpts1
        }
