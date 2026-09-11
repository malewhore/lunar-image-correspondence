import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.matching.matcher import DeepMatcher
from src.schemas.processed_pair import ProcessedPair
from src.ingestion.processed_pair_adapter import ProcessedPairAdapter
from src.evaluation.metrics import evaluate_pair

def read_pds4_crop(filepath, lines, samples, crop_size=1024):
    """Reads a center crop from a massive PDS4 raw .img file."""
    try:
        # Calculate offset to the center
        start_line = lines // 2 - crop_size // 2
        start_sample = samples // 2 - crop_size // 2
        
        crop = np.zeros((crop_size, crop_size), dtype=np.uint8)
        
        with open(filepath, "rb") as f:
            for i in range(crop_size):
                line_idx = start_line + i
                f.seek(line_idx * samples + start_sample)
                line_data = f.read(crop_size)
                if line_data:
                    crop[i, :len(line_data)] = np.frombuffer(line_data, dtype=np.uint8)
        return crop
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        return None

def main():
    print("================================================")
    print("Baseline Evaluation: Full Frozen Set")
    print("================================================")

    matcher = DeepMatcher(max_keypoints=2048)
    adapter = ProcessedPairAdapter()
    
    pairs_to_test = [
        {
            "id": "Pair_01",
            "ref": "data/real_pairs_drive/Reference KAGGLE",
            "mov": "data/real_pairs_drive/MOVING KAGGLE",
            "type": "image"
        },
        {
            "id": "Pair_02",
            "ref": "data/real_pairs_drive/pair_002/reference/quickmap-lroc.png",
            "mov": "data/real_pairs_drive/pair_002/moving/data/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_d_img_d18.img",
            "type": "pds4",
            "lines": 101075,
            "samples": 12000
        }
    ]
    
    print("| Pair | Model | Candidate Matches | Inliers | Inlier Ratio | Runtime |")
    print("|---|---|---|---|---|---|")
    
    for p in pairs_to_test:
        if p["type"] == "image":
            ref_img = cv2.imread(p["ref"], cv2.IMREAD_GRAYSCALE)
            mov_img = cv2.imread(p["mov"], cv2.IMREAD_GRAYSCALE)
        else:
            ref_img = cv2.imread(p["ref"], cv2.IMREAD_GRAYSCALE)
            mov_img = read_pds4_crop(p["mov"], p["lines"], p["samples"])
            
        if ref_img is None or mov_img is None:
            print(f"| {p['id']} | Pretrained SP+LG | Error | Error | Error | Error |")
            continue
            
        # Optional: scale down if massive
        if ref_img.shape[0] > 2000 or ref_img.shape[1] > 2000:
            scale = 1000 / max(ref_img.shape)
            ref_img = cv2.resize(ref_img, (0,0), fx=scale, fy=scale)
            
        if mov_img.shape[0] > 2000 or mov_img.shape[1] > 2000:
            scale = 1000 / max(mov_img.shape)
            mov_img = cv2.resize(mov_img, (0,0), fx=scale, fy=scale)

        pair = ProcessedPair(pair_id=p["id"], reference=ref_img, moving=mov_img)
        input_dict = adapter.adapt(pair)
        
        t0 = time.time()
        results = matcher.extract_and_match(input_dict["image0"], input_dict["image1"])
        t1 = time.time()
        
        metrics = evaluate_pair(results, input_dict["image0"].shape[-2:], input_dict["image1"].shape[-2:])
        
        c_matches = metrics["candidate_matches"]
        inliers = metrics["ransac_inliers"]
        ratio = metrics["inlier_ratio"]
        runtime = t1 - t0
        
        print(f"| {p['id']} | Pretrained SP+LG | {c_matches} | {inliers} | {ratio:.2f}% | {runtime:.2f}s |")

if __name__ == "__main__":
    main()
