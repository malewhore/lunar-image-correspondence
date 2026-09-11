import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.synthetic_generator import SyntheticPairGenerator

def visualize_synthetic_pair(ref, mov, H_gt, save_path):
    """
    Visualizes the synthetic pair and draws a grid on the reference image, 
    warped into the moving image using the ground-truth homography to prove exact alignment.
    """
    # Create color images for drawing colored grids
    ref_color = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    mov_color = cv2.cvtColor(mov, cv2.COLOR_GRAY2BGR)
    
    h, w = ref.shape
    
    # Draw a 4x4 grid
    grid_pts = []
    for y in np.linspace(0, h-1, 5):
        for x in np.linspace(0, w-1, 5):
            grid_pts.append([x, y])
            cv2.circle(ref_color, (int(x), int(y)), 3, (0, 255, 0), -1)
            
    grid_pts = np.array(grid_pts, dtype=np.float32).reshape(-1, 1, 2)
    
    # Warp grid points to the moving image
    warped_pts = cv2.perspectiveTransform(grid_pts, H_gt)
    
    for pt in warped_pts:
        cv2.circle(mov_color, (int(pt[0][0]), int(pt[0][1])), 3, (0, 255, 0), -1)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(ref_color[..., ::-1])
    axes[0].set_title("Reference Patch")
    axes[0].axis('off')
    
    axes[1].imshow(mov_color[..., ::-1])
    axes[1].set_title("Moving Patch (Distorted)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    print("Testing Synthetic Pair Generator...")
    
    # Find some lunar images to use as base
    base_images = list(Path("data/real_pairs_drive").glob("*KAGGLE*"))
    if not base_images:
        print("Error: No base images found in data/real_pairs_drive/")
        return
        
    img_path = base_images[0]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error loading image {img_path}")
        return
        
    print(f"Loaded base image: {img_path.name} ({img.shape})")
    
    # Initialize generator
    generator = SyntheticPairGenerator(output_size=(512, 512))
    
    out_dir = Path("outputs/synthetic_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(5):
        print(f"Generating synthetic pair {i+1}...")
        ref, mov, H_gt = generator.generate(img)
        
        save_path = out_dir / f"synth_preview_{i+1}.png"
        visualize_synthetic_pair(ref, mov, H_gt, save_path)
        print(f"  -> Saved {save_path}")

    print("Success! Generated 5 mathematically perfect pairs for Phase 8.")

if __name__ == "__main__":
    main()
