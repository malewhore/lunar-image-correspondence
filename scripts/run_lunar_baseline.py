import sys
import time
import cv2
from pathlib import Path

# Add root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.matching.matcher import DeepMatcher
from src.schemas.processed_pair import PreprocessedPair
from src.ingestion.processed_pair_adapter import PreprocessedPairAdapter
from src.evaluation.metrics import BaselineMetrics
from lightglue import viz2d

def main():
    print("================================================")
    print("Lunar Baseline: SuperPoint + LightGlue")
    print("================================================")

    # 1. Mock the input PreprocessedPair
    # Load our demo images as grayscale uint8 numpy arrays to simulate the preprocessing output
    demo_ref = "data/real_pairs_drive/Reference KAGGLE"
    demo_mov = "data/real_pairs_drive/MOVING KAGGLE"
    
    if not Path(demo_ref).exists() or not Path(demo_mov).exists():
        print(f"Error: Missing mock lunar images in data/real_pairs/")
        return

    ref_img = cv2.imread(demo_ref, cv2.IMREAD_GRAYSCALE)
    mov_img = cv2.imread(demo_mov, cv2.IMREAD_GRAYSCALE)

    pair = PreprocessedPair(
        pair_id="demo_lunar_001",
        representation="grayscale",
        reference=ref_img,
        moving=mov_img,
        metadata={
            "reference_sensor": "LROC",
            "moving_sensor": "OHRC",
            "resolution": "mixed"
        }
    )
    
    print(f"Ingesting PreprocessedPair:")
    print(f"  Pair ID : {pair.pair_id}")
    print(f"  Ref Shape: {pair.reference.shape}")
    print(f"  Mov Shape: {pair.moving.shape}")

    # 2. Adapt the input
    adapter = PreprocessedPairAdapter()
    image0, image1 = adapter.adapt(pair)

    # 3. Initialize Matcher
    print("\nInitializing models...")
    matcher = DeepMatcher(max_keypoints=2048)

    # 4. Extract and Match
    print("Extracting and matching features...")
    start_time = time.time()
    results = matcher.extract_and_match(image0, image1)
    inference_time = time.time() - start_time

    # 5. Compute Metrics
    print("\nComputing baseline metrics...")
    metrics = BaselineMetrics.compute_metrics(
        matches=results['matches'].cpu().numpy(),
        kpts0=results['m_kpts0'].cpu().numpy(),
        kpts1=results['m_kpts1'].cpu().numpy(),
        inference_time=inference_time
    )

    print(f"\nResults:")
    print(f"  Candidate Matches : {metrics['candidate_matches']}")
    print(f"  RANSAC Inliers    : {metrics['inliers']}")
    print(f"  Inlier Ratio      : {metrics['inlier_ratio']:.2%}")
    print(f"  Inference Time    : {metrics['inference_time']:.3f} s")

    # 6. Visualize
    print("\nSaving visualizations...")
    axes = viz2d.plot_images([image0.squeeze(0).cpu(), image1.squeeze(0).cpu()])
    viz2d.plot_matches(results['m_kpts0'].cpu(), results['m_kpts1'].cpu(), color='lime', lw=0.2)
    
    out_dir = Path("outputs/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lunar_baseline_matches.png"
    viz2d.save_plot(str(out_path))

    print(f"\nStatus            : SUCCESS")
    print(f"Visualization saved to {out_path}")
    print("================================================")

if __name__ == "__main__":
    main()
