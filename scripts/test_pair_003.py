import sys
from pathlib import Path
import cv2
import numpy as np

# Add root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.matching.matcher import DeepMatcher
from src.schemas.processed_pair import PreprocessedPair
from src.ingestion.processed_pair_adapter import PreprocessedPairAdapter
from src.evaluation.metrics import BaselineMetrics

def read_memmap_crop(filepath, lines, samples, crop_size=1000):
    """Fast reading of PDS4 .img using memmap."""
    print(f"Mapping massive file: {filepath}")
    memmapped_array = np.memmap(filepath, dtype=np.uint8, mode='r', shape=(lines, samples))
    
    start_line = lines // 2 - crop_size // 2
    start_sample = samples // 2 - crop_size // 2
    
    print(f"Extracting {crop_size}x{crop_size} crop from center...")
    crop = memmapped_array[start_line:start_line+crop_size, start_sample:start_sample+crop_size].copy()
    
    # Clean up memmap
    del memmapped_array
    return crop

def main():
    print("================================================")
    print("Testing Baseline on PAIR_003")
    print("================================================")

    # 1. Paths
    ref_path = "data/real_pairs_drive/pair_003/reference/quickmap-lroc.png"
    mov_path = "data/real_pairs_drive/pair_003/moving/data/calibrated/20210402/ch2_ohr_ncp_20210402T0546284043_d_img_d18.img"
    
    # 2. Dimensions from XML
    lines = 78175
    samples = 12000
    
    # 3. Load Images
    print(f"Loading reference: {ref_path}")
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        print("Failed to load reference image.")
        return
        
    mov_img = read_memmap_crop(mov_path, lines, samples, crop_size=2000)
    
    # 4. Downscale for memory/time constraints during matching
    if ref_img.shape[0] > 1000 or ref_img.shape[1] > 1000:
        scale = 1000 / max(ref_img.shape)
        ref_img = cv2.resize(ref_img, (0,0), fx=scale, fy=scale)
        
    if mov_img.shape[0] > 1000 or mov_img.shape[1] > 1000:
        scale = 1000 / max(mov_img.shape)
        mov_img = cv2.resize(mov_img, (0,0), fx=scale, fy=scale)

    print(f"Reference shape: {ref_img.shape}")
    print(f"Moving crop shape: {mov_img.shape}")

    # 5. Pipeline
    print("Initializing SuperPoint + LightGlue...")
    matcher = DeepMatcher(max_keypoints=2048)
    adapter = PreprocessedPairAdapter()
    
    pair = PreprocessedPair(pair_id="Pair_003", representation="image", reference=ref_img, moving=mov_img, metadata={})
    image0, image1 = adapter.adapt(pair)
    
    print("Extracting & Matching...")
    results = matcher.extract_and_match(image0, image1)
    
    print("Evaluating metrics...")
    metrics = BaselineMetrics.compute_metrics(
        matches=results['matches'].cpu().numpy(),
        kpts0=results['m_kpts0'].cpu().numpy(),
        kpts1=results['m_kpts1'].cpu().numpy(),
        inference_time=0.0
    )
    
    c_matches = metrics["candidate_matches"]
    inliers = metrics["inliers"]
    ratio = metrics["inlier_ratio"]
    
    print("================================================")
    print(f"RESULTS FOR PAIR_003:")
    print(f"Candidate Matches: {c_matches}")
    print(f"RANSAC Inliers:    {inliers}")
    print(f"Inlier Ratio:      {ratio:.2f}%")
    print("================================================")

if __name__ == "__main__":
    main()
