import sys
from pathlib import Path
import cv2
import numpy as np
import torch

# Add root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.matching.matcher import DeepMatcher
from src.schemas.processed_pair import PreprocessedPair
from src.ingestion.processed_pair_adapter import PreprocessedPairAdapter
from src.evaluation.metrics import BaselineMetrics

def read_memmap_crop(filepath, lines, samples, crop_size=1000):
    memmapped_array = np.memmap(filepath, dtype=np.uint8, mode='r', shape=(lines, samples))
    start_line = lines // 2 - crop_size // 2
    start_sample = samples // 2 - crop_size // 2
    crop = memmapped_array[start_line:start_line+crop_size, start_sample:start_sample+crop_size].copy()
    del memmapped_array
    return crop

def main():
    print("================================================")
    print("Phase 12: Evaluating on Real Lunar Pairs")
    print("================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load images for Pair 002
    ref_path = "data/real_pairs_drive/pair_002/reference/quickmap-lroc.png"
    mov_path = "data/real_pairs_drive/pair_002/moving/data/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_d_img_d18.img"
    
    lines = 101075
    samples = 12000
    
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        print("Failed to load reference image.")
        return
        
    mov_img = read_memmap_crop(mov_path, lines, samples, crop_size=2000)
    
    # Downscale for memory
    if ref_img.shape[0] > 1000 or ref_img.shape[1] > 1000:
        scale = 1000 / max(ref_img.shape)
        ref_img = cv2.resize(ref_img, (0,0), fx=scale, fy=scale)
        
    if mov_img.shape[0] > 1000 or mov_img.shape[1] > 1000:
        scale = 1000 / max(mov_img.shape)
        mov_img = cv2.resize(mov_img, (0,0), fx=scale, fy=scale)

    adapter = PreprocessedPairAdapter()
    pair = PreprocessedPair(pair_id="Pair_002", representation="image", reference=ref_img, moving=mov_img, metadata={})
    image0, image1 = adapter.adapt(pair)
    
    # --- PRETRAINED EVALUATION ---
    print("\n--- Evaluating Pretrained Baseline ---")
    pretrained_matcher = DeepMatcher(max_keypoints=2048)
    pretrained_matcher.extractor = pretrained_matcher.extractor.to(device)
    pretrained_matcher.matcher = pretrained_matcher.matcher.to(device)
    
    results_pre = pretrained_matcher.extract_and_match(image0, image1)
    metrics_pre = BaselineMetrics.compute_metrics(
        matches=results_pre['matches'].cpu().numpy(),
        kpts0=results_pre['m_kpts0'].cpu().numpy(),
        kpts1=results_pre['m_kpts1'].cpu().numpy(),
        inference_time=0.0
    )
    
    # --- FINE-TUNED EVALUATION ---
    print("\n--- Evaluating Fine-Tuned Model ---")
    finetuned_matcher = DeepMatcher(max_keypoints=2048)
    finetuned_matcher.extractor = finetuned_matcher.extractor.to(device)
    finetuned_matcher.matcher = finetuned_matcher.matcher.to(device)
    
    ckpt_path = Path("outputs/checkpoints/lightglue_lunar_best.pth")
    if ckpt_path.exists():
        finetuned_matcher.matcher.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        results_ft = finetuned_matcher.extract_and_match(image0, image1)
        metrics_ft = BaselineMetrics.compute_metrics(
            matches=results_ft['matches'].cpu().numpy(),
            kpts0=results_ft['m_kpts0'].cpu().numpy(),
            kpts1=results_ft['m_kpts1'].cpu().numpy(),
            inference_time=0.0
        )
    else:
        print(f"Checkpoint {ckpt_path} not found!")
        metrics_ft = {"candidate_matches": 0, "inliers": 0, "inlier_ratio": 0.0}
        
    print("\n================================================")
    print("COMPARISON (Real Pair 002)")
    print(f"Metric\t\t\tPretrained\tFine-Tuned")
    print(f"Candidate Matches\t{metrics_pre['candidate_matches']}\t\t{metrics_ft['candidate_matches']}")
    print(f"RANSAC Inliers\t\t{metrics_pre['inliers']}\t\t{metrics_ft['inliers']}")
    print(f"Inlier Ratio\t\t{metrics_pre['inlier_ratio']:.2f}%\t\t{metrics_ft['inlier_ratio']:.2f}%")
    print("================================================")

if __name__ == "__main__":
    main()
