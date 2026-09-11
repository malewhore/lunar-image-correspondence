import numpy as np
import cv2

class SyntheticHomographyGenerator:
    """
    Generates realistic 3x3 homography matrices for synthetic dataset generation.
    It simulates scaling, rotation, translation, and perspective warps.
    """
    def __init__(self, 
                 scale_range=(0.8, 1.2),
                 rotation_range=(-15, 15), # in degrees
                 translation_range=(-0.1, 0.1), # as a fraction of image size
                 perspective_range=(-0.0005, 0.0005)):
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.perspective_range = perspective_range

    def generate(self, image_shape):
        """
        Generates a 3x3 homography matrix.
        image_shape: (H, W) or (H, W, C)
        """
        H, W = image_shape[:2]
        center = (W / 2.0, H / 2.0)

        # 1. Random Scale & Rotation
        scale = np.random.uniform(*self.scale_range)
        angle = np.random.uniform(*self.rotation_range)
        
        # 2x3 Affine matrix for Scale + Rotation
        M_rot_scale = cv2.getRotationMatrix2D(center, angle, scale)
        
        # 2. Random Translation
        tx = np.random.uniform(*self.translation_range) * W
        ty = np.random.uniform(*self.translation_range) * H
        M_rot_scale[0, 2] += tx
        M_rot_scale[1, 2] += ty

        # Convert to 3x3 Homography
        H_mat = np.eye(3, dtype=np.float32)
        H_mat[0:2, :] = M_rot_scale

        # 3. Random Perspective Warp (bottom row of homography)
        p1 = np.random.uniform(*self.perspective_range)
        p2 = np.random.uniform(*self.perspective_range)
        
        H_persp = np.eye(3, dtype=np.float32)
        H_persp[2, 0] = p1
        H_persp[2, 1] = p2
        
        # Combine
        H_final = H_persp @ H_mat
        
        # Normalize
        H_final /= H_final[2, 2]
        
        return H_final

    def warp_image(self, image, H_mat):
        """
        Warps the image using the given homography matrix.
        """
        H, W = image.shape[:2]
        warped = cv2.warpPerspective(image, H_mat, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return warped
