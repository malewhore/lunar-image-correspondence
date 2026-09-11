import numpy as np
import cv2
import random

class SyntheticPairGenerator:
    def __init__(self, output_size=(512, 512), max_rotation=15, max_scale=0.15, max_perspective=0.1):
        """
        Generates synthetic moving/reference pairs by applying a random homography 
        to crops of a single large lunar image.
        """
        self.output_size = output_size
        self.max_rotation = max_rotation
        self.max_scale = max_scale
        self.max_perspective = max_perspective

    def apply_photometric_distortion(self, img):
        """Apply random brightness, contrast, and noise to simulate sensor differences."""
        # Add Gaussian noise
        noise = np.random.normal(0, 5, img.shape).astype(np.float32)
        img = cv2.add(img.astype(np.float32), noise)
        
        # Contrast & Brightness
        alpha = random.uniform(0.8, 1.2) # Contrast
        beta = random.uniform(-15, 15)   # Brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img

    def generate(self, image):
        """
        Takes a base image and returns:
        - ref_patch: (H, W) uint8 reference patch
        - mov_patch: (H, W) uint8 moving patch (distorted)
        - H_gt: (3, 3) Ground-Truth homography mapping ref_patch coords to mov_patch coords
        """
        h, w = image.shape[:2]
        ref_h, ref_w = self.output_size
        
        # Ensure image is large enough
        min_size = int(max(self.output_size) * 1.5)
        if h < min_size or w < min_size:
            image = cv2.resize(image, (max(w, min_size), max(h, min_size)))
            h, w = image.shape[:2]

        # 1. Randomly crop a larger working area
        crop_size = min_size
        y1 = random.randint(0, h - crop_size)
        x1 = random.randint(0, w - crop_size)
        base_patch = image[y1:y1+crop_size, x1:x1+crop_size]

        # The reference patch is taken from the exact center of this working area
        center_y, center_x = crop_size // 2, crop_size // 2
        ref_y1 = center_y - ref_h // 2
        ref_x1 = center_x - ref_w // 2
        ref_patch = base_patch[ref_y1:ref_y1+ref_h, ref_x1:ref_x1+ref_w]

        # 2. Generate random perspective transformation
        # Define the 4 corners of the reference patch inside the base_patch coordinate system
        src_pts = np.float32([
            [ref_x1, ref_y1],
            [ref_x1 + ref_w, ref_y1],
            [ref_x1 + ref_w, ref_y1 + ref_h],
            [ref_x1, ref_y1 + ref_h]
        ])

        # Perturb the corners for perspective distortion
        dst_pts = src_pts.copy()
        for i in range(4):
            dst_pts[i][0] += random.uniform(-self.max_perspective, self.max_perspective) * ref_w
            dst_pts[i][1] += random.uniform(-self.max_perspective, self.max_perspective) * ref_h

        # Add global rotation and scale
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        scale = random.uniform(1.0 - self.max_scale, 1.0 + self.max_scale)
        M_rot = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
        
        # Apply rotation and scale to the destination points
        ones = np.ones((4, 1))
        dst_pts_hom = np.hstack([dst_pts, ones])
        dst_pts = M_rot.dot(dst_pts_hom.T).T

        # Compute Homography from source coordinates to distorted coordinates
        H_base_to_warped = cv2.getPerspectiveTransform(src_pts, dst_pts.astype(np.float32))

        # Warp the base patch
        warped_base = cv2.warpPerspective(
            base_patch, 
            H_base_to_warped, 
            (crop_size, crop_size), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REFLECT_101
        )

        # The moving patch is the same center crop taken from the warped base
        mov_patch = warped_base[ref_y1:ref_y1+ref_h, ref_x1:ref_x1+ref_w]

        # Apply photometric distortions only to moving patch
        mov_patch = self.apply_photometric_distortion(mov_patch)

        # 3. Compute mathematically exact ground-truth homography
        # Map: [ref_patch local] -> [base_patch] -> [warped_base] -> [mov_patch local]
        T_center = np.array([
            [1, 0, ref_x1],
            [0, 1, ref_y1],
            [0, 0, 1]
        ], dtype=np.float32)

        T_center_inv = np.array([
            [1, 0, -ref_x1],
            [0, 1, -ref_y1],
            [0, 0, 1]
        ], dtype=np.float32)

        H_gt = T_center_inv @ H_base_to_warped @ T_center

        return ref_patch, mov_patch, H_gt
