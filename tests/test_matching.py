import sys
import torch
from pathlib import Path

# Add root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.matching.matcher import DeepMatcher

def test_matcher_initialization():
    matcher = DeepMatcher(max_keypoints=512)
    assert matcher is not None
    assert matcher.extractor is not None
    assert matcher.matcher is not None

def test_extract_and_match():
    # Create dummy images [C, H, W]
    img0 = torch.rand(3, 256, 256)
    img1 = torch.rand(3, 256, 256)
    
    matcher = DeepMatcher(max_keypoints=100)
    results = matcher.extract_and_match(img0, img1)
    
    assert 'kpts0' in results
    assert 'kpts1' in results
    assert 'matches' in results
    assert 'm_kpts0' in results
    assert 'm_kpts1' in results

if __name__ == "__main__":
    print("Running tests...")
    test_matcher_initialization()
    test_extract_and_match()
    print("All tests passed!")
