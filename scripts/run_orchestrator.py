import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.matching.matcher import DeepMatcher
from src.schemas.processed_pair import PreprocessedPair
from src.ingestion.processed_pair_adapter import PreprocessedPairAdapter
from src.matching.adapter import AlgorithmAdapter

def read_memmap_crop(filepath, lines, samples, crop_size=1000):
    """Phase 1-2: Raw image ingestion and extraction"""
    memmapped_array = np.memmap(filepath, dtype=np.uint8, mode='r', shape=(lines, samples))
    start_line = lines // 2 - crop_size // 2
    start_sample = samples // 2 - crop_size // 2
    crop = memmapped_array[start_line:start_line+crop_size, start_sample:start_sample+crop_size].copy()
    del memmapped_array
    return crop

def draw_matches(img0, img1, kpts0, kpts1, mask, save_path):
    """Visualizes the geometric inliers."""
    # Convert points to integers for drawing
    kpts0_int = np.round(kpts0).astype(int)
    kpts1_int = np.round(kpts1).astype(int)
    
    # Filter inliers
    inliers0 = kpts0_int[mask]
    inliers1 = kpts1_int[mask]

    # Create a side-by-side visualization image
    h0, w0 = img0.shape
    h1, w1 = img1.shape
    vis = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    
    # Convert grayscale to BGR for visualization
    vis[:h0, :w0, :] = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    vis[:h1, w0:w0+w1, :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    
    # Draw matches
    for pt1, pt2 in zip(inliers0, inliers1):
        x1, y1 = pt1
        x2, y2 = pt2[0] + w0, pt2[1]
        
        cv2.circle(vis, (x1, y1), 5, (0, 255, 0), -1)
        cv2.circle(vis, (x2, y2), 5, (0, 255, 0), -1)
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    cv2.imwrite(str(save_path), vis)
    print(f"Visualization saved to {save_path}")

def main():
    print("================================================")
    print("PHASE 15: REGISTRATION ORCHESTRATOR END-TO-END")
    print("================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. INGESTION
    print("\n[1/4] Ingesting raw lunar imagery...")
    ref_path = "data/real_pairs_drive/pair_001/reference/quickmap-lroc.png"
    mov_path = "data/real_pairs_drive/pair_001/moving/data/calibrated/20241115/ch2_ohr_ncp_20241115T1525004388_d_img_d18.img"
    
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    mov_img = read_memmap_crop(mov_path, lines=101074, samples=12000, crop_size=2000)
    
    # Downscale for memory
    if ref_img.shape[0] > 1000 or ref_img.shape[1] > 1000:
        scale = 1000 / max(ref_img.shape)
        ref_img = cv2.resize(ref_img, (0,0), fx=scale, fy=scale)
        
    if mov_img.shape[0] > 1000 or mov_img.shape[1] > 1000:
        scale = 1000 / max(mov_img.shape)
        mov_img = cv2.resize(mov_img, (0,0), fx=scale, fy=scale)

    # 2. PREPROCESSING CONTRACT
    print("[2/4] Formatting via PreprocessedPair Schema...")
    adapter = PreprocessedPairAdapter()
    pair = PreprocessedPair(pair_id="Pair_001", representation="image", reference=ref_img, moving=mov_img, metadata={})
    
    # 3. ALGORITHM ORCHESTRATION
    print("[3/4] Routing through AlgorithmAdapter (SuperPoint+LightGlue)...")
    pretrained_matcher = DeepMatcher(max_keypoints=2048)
    pretrained_matcher.extractor = pretrained_matcher.extractor.to(device)
    pretrained_matcher.matcher = pretrained_matcher.matcher.to(device)
    
    algo_adapter = AlgorithmAdapter(pretrained_matcher)
    
    # This single call encompasses Extraction, Matching, and RANSAC Verification
    result = algo_adapter.process(pair)
    
    # 4. STANDARDIZED RESULTS
    print("\n[4/4] AlgorithmResult Contract Output:")
    print(f"Pair ID: {result.pair_id}")
    print(f"Algorithm: {result.algorithm_name}")
    print(f"Candidate Matches: {result.metrics['candidate_matches']}")
    print(f"RANSAC Inliers: {result.metrics['inliers']}")
    print(f"Inlier Ratio: {result.metrics['inlier_ratio']:.2%}")
    print(f"Inference Time: {result.metrics['inference_time']:.2f} seconds")
    
    if result.transformation_matrix is not None:
        print("\nHomography Matrix:")
        print(np.array_str(result.transformation_matrix, precision=4, suppress_small=True))
    else:
        print("\nWARNING: Failed to compute Homography.")
        
    # Visual validation
    out_dir = Path("outputs/results")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "pair_001_final_registration.png"
    
    if result.metrics['inliers'] > 0:
        # Note: AlgoAdapter expects PreprocessedPair adapter arrays.
        img0_tensor, img1_tensor = adapter.adapt(pair)
        img0 = (img0_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        img1 = (img1_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        draw_matches(img0, img1, 
                     result.matched_kpts_moving, result.matched_kpts_reference, 
                     result.inlier_mask, out_path)
        
    print("\nOrchestrator Execution Complete.")

if __name__ == "__main__":
    main()
