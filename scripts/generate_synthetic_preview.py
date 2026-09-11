import sys
import torch
import cv2
import numpy as np
from pathlib import Path

# Add root to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.lunar_dataset import LunarSyntheticDataset
from lightglue import viz2d

def main():
    print("================================================")
    print("Generating Synthetic Previews (Phase 9)")
    print("================================================")
    
    data_dir = "data/real_pairs_drive"
    out_dir = Path("outputs/synthetic_previews")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = LunarSyntheticDataset(data_dir=data_dir, crop_size=(1024, 1024), length=5)
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        return

    for i in range(5):
        data = dataset[i]
        
        # Output tensors are [1, H, W] in range [0, 1]
        img0_tensor = data["image0"]
        img1_tensor = data["image1"]
        H_gt = data["homography"].numpy()
        
        print(f"Pair {i+1}:")
        print(f"  Image0 shape: {img0_tensor.shape}")
        print(f"  Image1 shape: {img1_tensor.shape}")
        print(f"  Ground Truth H:\n{H_gt}")
        
        # To verify the homography, let's draw a grid on Image0 and warp it to Image1
        H, W = img0_tensor.shape[1], img0_tensor.shape[2]
        
        # Convert tensors back to numpy for visualization
        img0_np = (img0_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img1_np = (img1_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        
        # Create a blank image with a grid
        grid = np.zeros((H, W), dtype=np.uint8)
        step = 128
        for x in range(0, W, step):
            cv2.line(grid, (x, 0), (x, H), 255, 2)
        for y in range(0, H, step):
            cv2.line(grid, (0, y), (W, y), 255, 2)
            
        # Warp the grid to the moving image frame
        warped_grid = cv2.warpPerspective(grid, H_gt, (W, H))
        
        # Overlay grid on images
        img0_color = cv2.cvtColor(img0_np, cv2.COLOR_GRAY2BGR)
        img1_color = cv2.cvtColor(img1_np, cv2.COLOR_GRAY2BGR)
        
        img0_color[grid > 0] = [0, 255, 0] # Green grid on reference
        img1_color[warped_grid > 0] = [0, 0, 255] # Red grid on moving
        
        # Concatenate horizontally
        combined = np.hstack((img0_color, img1_color))
        
        out_path = out_dir / f"synthetic_pair_{i:03d}.png"
        cv2.imwrite(str(out_path), combined)
        print(f"  Saved preview to: {out_path}\n")

    print("Success! Previews generated.")

if __name__ == "__main__":
    main()
