import numpy as np
import cv2

class SyntheticPhotometricDistortion:
    """
    Applies lunar-specific photometric distortions to an image to simulate
    the differences between reference and moving sensors.
    """
    def __init__(self,
                 brightness_range=(-0.2, 0.2),
                 contrast_range=(0.8, 1.2),
                 noise_std_range=(0.0, 10.0),
                 apply_shadows=True):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std_range = noise_std_range
        self.apply_shadows = apply_shadows

    def _apply_terminator_gradient(self, image):
        """
        Simulates a lighting gradient (like terminator line shadow fading).
        """
        H, W = image.shape[:2]
        
        # Create a gradient image
        gradient = np.linspace(0.4, 1.0, W, dtype=np.float32)
        # Randomly flip the gradient direction
        if np.random.rand() > 0.5:
            gradient = gradient[::-1]
            
        gradient_2d = np.tile(gradient, (H, 1))
        
        # Apply it
        img_f = image.astype(np.float32) * gradient_2d
        return np.clip(img_f, 0, 255).astype(np.uint8)

    def __call__(self, image):
        """
        Applies random distortions.
        """
        img = image.astype(np.float32)
        
        # 1. Random Brightness & Contrast
        brightness = np.random.uniform(*self.brightness_range) * 255.0
        contrast = np.random.uniform(*self.contrast_range)
        
        img = img * contrast + brightness
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # 2. Terminator/Shadow Gradient
        if self.apply_shadows and np.random.rand() > 0.5:
            img = self._apply_terminator_gradient(img)
            
        # 3. Additive Gaussian Sensor Noise
        noise_std = np.random.uniform(*self.noise_std_range)
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, img.shape)
            img_f = img.astype(np.float32) + noise
            img = np.clip(img_f, 0, 255).astype(np.uint8)
            
        return img
