import sys
import time
from pathlib import Path

# Add root to Python path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.matching.matcher import DeepMatcher
from lightglue.utils import load_image
from lightglue import viz2d

def main():
    print("================================================")
    print("SuperPoint + LightGlue Baseline")
    print("================================================")

    img1_path = Path("data/demo/image1.jpg")
    img2_path = Path("data/demo/image2.jpg")

    print(f"\nImage 1: {img1_path.name}")
    print(f"Image 2: {img2_path.name}")

    if not img1_path.exists() or not img2_path.exists():
        print("Error: Demo images not found in data/demo/")
        return

    # Initialize matcher
    print("Initializing models...")
    matcher = DeepMatcher(max_keypoints=2048)

    # Load images
    print("Loading images...")
    image0 = load_image(str(img1_path))
    image1 = load_image(str(img2_path))

    # Match
    print("Extracting and matching features...")
    start_time = time.time()
    results = matcher.extract_and_match(image0, image1)
    inference_time = time.time() - start_time

    kpts0 = results['kpts0']
    kpts1 = results['kpts1']
    matches = results['matches']

    print(f"\nKeypoints Image 1 : {len(kpts0)}")
    print(f"Keypoints Image 2 : {len(kpts1)}")
    print(f"Candidate Matches : {len(matches)}")
    print(f"Inference Time    : {inference_time:.3f} seconds")

    # Visualization
    print("\nVisualizing matches...")
    axes = viz2d.plot_images([image0.cpu(), image1.cpu()])
    viz2d.plot_matches(results['m_kpts0'].cpu(), results['m_kpts1'].cpu(), color='lime', lw=0.2)
    
    out_dir = Path("outputs/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ordinary_matches.png"
    viz2d.save_plot(str(out_path))

    print(f"\nStatus            : SUCCESS")
    print(f"Visualization saved to {out_path}")
    print("================================================")

if __name__ == "__main__":
    main()
