import sys
from pathlib import Path
import torch
import numpy as np
from src.schemas.processed_pair import PreprocessedPair

# Add temp_repo to path so we can use their preprocessing module
TEMP_REPO_SRC = Path(__file__).resolve().parent.parent.parent / "data" / "temp_repo" / "src"
if str(TEMP_REPO_SRC) not in sys.path:
    sys.path.append(str(TEMP_REPO_SRC))

from preprocessing.preprocess import create_valid_mask, robust_normalize, apply_clahe, structural_image

class PreprocessedPairAdapter:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _apply_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """Applies the mandatory classical CV preprocessing algorithm."""
        # Ensure grayscale
        if img.ndim == 3:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        mask = create_valid_mask(img)
        normalized = robust_normalize(img, mask, low_percentile=1.0, high_percentile=99.0)
        enhanced = apply_clahe(normalized, clip_limit=2.0, tile_grid_size=(8, 8))
        structural = structural_image(enhanced, gradient_weight=0.30)
        
        return structural
        
    def adapt(self, pair: PreprocessedPair):
        """
        Converts the PreprocessedPair numpy arrays into tensors expected by DeepMatcher.
        Mandatory preprocessing is applied to normalize lunar illumination.
        Returns image0 (reference) and image1 (moving).
        SuperPoint/LightGlue expects [1, 1, H, W] float32 tensors scaled to [0, 1].
        """
        def to_tensor(img: np.ndarray):
            # 1. Apply mandatory preprocessing
            processed_img = self._apply_preprocessing(img)
            
            # 2. Ensure 3D (C, H, W) where C=1
            if processed_img.ndim == 2:
                processed_img = processed_img[None, ...]
            # 3. Convert to float and scale to [0, 1] (preprocessing returns uint8 0-255)
            t = torch.from_numpy(processed_img).float() / 255.0
            # 4. Add batch dimension
            t = t.unsqueeze(0)
            return t.to(self.device)

        image0 = to_tensor(pair.reference)
        image1 = to_tensor(pair.moving)
        return image0, image1
