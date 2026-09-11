"""Pretrained SuperPoint + LightGlue baseline (no fine-tuning)."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from backend.matchers.base import MatcherInterface
from backend.schemas.matching import MatchingResult
from backend.schemas.processed_pair import ProcessedPair


class LightGlueBaselineMatcher(MatcherInterface):
    """
    Official pretrained SuperPoint + LightGlue.
    Does not modify weights. Accepts ProcessedPair only.
    """

    def __init__(
        self,
        max_num_keypoints: int = 2048,
        filter_threshold: float = 0.1,
        device: Optional[str] = None,
    ):
        from lightglue import LightGlue, SuperPoint

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.extractor = (
            SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        )
        self.matcher = (
            LightGlue(features="superpoint", filter_threshold=filter_threshold)
            .eval()
            .to(self.device)
        )

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        """Convert uint8/float grayscale HxW → float tensor [1, 1, H, W] in [0, 1]."""
        if image.ndim == 3:
            image = image[:, :, 0]
        arr = image.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        tensor = torch.from_numpy(arr)[None, None]
        return tensor

    @torch.inference_mode()
    def match(self, pair: ProcessedPair) -> MatchingResult:
        from lightglue.utils import rbd

        start = time.time()
        image0 = self._to_tensor(pair.moving_image).to(self.device)
        image1 = self._to_tensor(pair.reference_image).to(self.device)

        feats0 = self.extractor.extract(image0)
        feats1 = self.extractor.extract(image1)
        matches01 = self.matcher({"image0": feats0, "image1": feats1})

        feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]

        kpts0 = feats0["keypoints"].detach().cpu().numpy()
        kpts1 = feats1["keypoints"].detach().cpu().numpy()
        matches = matches01["matches"].detach().cpu().numpy()
        scores = matches01["scores"].detach().cpu().numpy()

        runtime = time.time() - start
        return MatchingResult(
            method="pretrained_superpoint_lightglue",
            keypoints_a=kpts0,
            keypoints_b=kpts1,
            matches=matches,
            confidence=scores,
            num_keypoints_a=len(kpts0),
            num_keypoints_b=len(kpts1),
            num_matches=len(matches),
            runtime_seconds=runtime,
        )
