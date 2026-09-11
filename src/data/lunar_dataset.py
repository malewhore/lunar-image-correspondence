import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from src.data.synthetic_homography import SyntheticHomographyGenerator
from src.data.synthetic_photometric import SyntheticPhotometricDistortion
from src.schemas.processed_pair import PreprocessedPair
from src.ingestion.processed_pair_adapter import PreprocessedPairAdapter

class LunarSyntheticDataset(Dataset):
    """
    A PyTorch dataset that takes raw lunar reference images, crops them,
    applies a synthetic homography and photometric distortion to generate
    a training pair, and passes it through the mandatory CV preprocessing.
    """
    def __init__(self, data_dir, crop_size=(1024, 1024), length=1000, device='cpu', split='train'):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.crop_size = crop_size
        self.length = length
        self.split = split
        
        # Discover base images (png or jpg)
        self.image_paths = list(self.data_dir.glob("**/*.png")) + list(self.data_dir.glob("**/*.jpg"))
        if not self.image_paths:
            raise ValueError(f"No PNG/JPG images found in {data_dir} to use as synthetic bases.")
            
        self.geom_generator = SyntheticHomographyGenerator()
        self.photo_generator = SyntheticPhotometricDistortion()
        
        # We need the adapter to apply CLAHE, normalize, and return standard 4D tensors
        self.adapter = PreprocessedPairAdapter(device=torch.device(device))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. Randomly pick a base image
        img_path = str(np.random.choice(self.image_paths))
        base_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. Extract a random crop from the base image
        H, W = base_img.shape
        ch, cw = self.crop_size
        
        if H <= ch or W <= cw:
            # Image is too small, resize it
            base_img = cv2.resize(base_img, (max(W, cw+100), max(H, ch+100)))
            H, W = base_img.shape
            
        # Geographic isolation:
        # Train: top 80%, Val: next 10%, Test: bottom 10%
        if self.split == 'train':
            y_min, y_max = 0, int(H * 0.8)
        elif self.split == 'val':
            y_min, y_max = int(H * 0.8), int(H * 0.9)
        else: # test
            y_min, y_max = int(H * 0.9), H
            
        # Ensure the region is large enough
        if y_max - y_min <= ch:
            y_min = max(0, y_max - ch - 10)
            
        y = np.random.randint(y_min, max(y_min + 1, y_max - ch))
        x = np.random.randint(0, W - cw)
        reference_crop = base_img[y:y+ch, x:x+cw]
        
        # 3. Generate ground truth homography
        H_gt = self.geom_generator.generate(self.crop_size)
        
        # 4. Warp reference crop to create "moving" crop
        moving_crop = self.geom_generator.warp_image(reference_crop, H_gt)
        
        # 5. Apply separate photometric distortions to both to simulate OHRC vs LROC
        reference_crop = self.photo_generator(reference_crop)
        moving_crop = self.photo_generator(moving_crop)
        
        # 6. Build the Domain object
        pair = PreprocessedPair(
            pair_id=f"synth_{idx}",
            representation="grayscale",
            reference=reference_crop,
            moving=moving_crop,
            metadata={"homography": H_gt}
        )
        
        # 7. Apply CV Preprocessing (CLAHE, Masking, etc.) to get network tensors
        # adapter.adapt returns 4D tensors: [1, 1, H, W]
        img0_tensor, img1_tensor = self.adapter.adapt(pair)
        
        # Drop the first batch dimension because DataLoader will add it back
        img0_tensor = img0_tensor.squeeze(0)  # [1, H, W]
        img1_tensor = img1_tensor.squeeze(0)  # [1, H, W]
        
        return {
            "image0": img0_tensor,
            "image1": img1_tensor,
            "homography": torch.from_numpy(H_gt).float()
        }
