import sys
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

# Add root to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.lunar_dataset import LunarSyntheticDataset
from src.matching.matcher import DeepMatcher
from src.evaluation.metrics import BaselineMetrics

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(matcher, dataloader, device):
    matcher.matcher.eval()
    
    total_precision = 0.0
    total_recall = 0.0
    total_rmse = 0.0
    total_inliers = 0
    count = 0
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            image0 = batch["image0"].to(device)
            image1 = batch["image1"].to(device)
            H_gt = batch["homography"].squeeze(0).cpu().numpy()
            
            # Forward pass
            results = matcher.extract_and_match(image0, image1)
            
            # Since this is synthetic, we have exact ground truth H to compute true precision/recall
            kpts0 = results['kpts0'].cpu().numpy()
            kpts1 = results['kpts1'].cpu().numpy()
            m_kpts0 = results['m_kpts0'].cpu().numpy()
            m_kpts1 = results['m_kpts1'].cpu().numpy()
            
            if len(kpts0) == 0 or len(kpts1) == 0:
                continue
            
            if len(m_kpts0) != len(m_kpts1):
                print(f"Shape mismatch! {m_kpts0.shape} vs {m_kpts1.shape}")
                continue
                
            metrics = BaselineMetrics.compute_metrics(
                matches=results['matches'].cpu().numpy(),
                kpts0=m_kpts0,
                kpts1=m_kpts1,
                inference_time=0.0
            )
            
            # For exact synthetic evaluation, we could warp all kpts0 by H_gt to see if they land on kpts1
            # But the baseline metrics already gives us RANSAC inliers.
            # RANSAC is a geometric proxy, let's also compute exact RMSE on inliers.
            m_kpts0 = results['m_kpts0'].cpu().numpy()
            m_kpts1 = results['m_kpts1'].cpu().numpy()
            inliers = metrics["inliers"]
            if inliers > 0 and len(m_kpts0) > 0:
                # Warp m_kpts0 using H_gt
                kpts0_h = np.concatenate([m_kpts0, np.ones((len(m_kpts0), 1))], axis=1)
                warped = (H_gt @ kpts0_h.T).T
                warped = warped[:, :2] / (warped[:, 2:] + 1e-8)
                
                # RMSE error
                rmse = np.sqrt(np.mean(np.sum((warped - m_kpts1)**2, axis=1)))
                
                # Precision/Recall approximations
                # Precision = Inliers / Candidate Matches
                precision = inliers / max(1, metrics["candidate_matches"])
                
                # Recall = Inliers / Total keypoints extracted (proxy)
                recall = inliers / 1024.0 # Max keypoints
                
                total_precision += precision
                total_recall += recall
                total_rmse += rmse
                total_inliers += inliers
                count += 1
                
    if count == 0:
        return {"precision": 0, "recall": 0, "rmse": 0, "avg_inliers": 0}
        
    return {
        "precision": total_precision / count,
        "recall": total_recall / count,
        "rmse": total_rmse / count,
        "avg_inliers": total_inliers / count
    }

def main():
    print("================================================")
    print("Phase 12: Evaluating on Synthetic Test Set")
    print("================================================")
    
    config = load_config("configs/finetune.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_dataset = LunarSyntheticDataset(
        data_dir=config['dataset']['data_dir'],
        crop_size=tuple(config['dataset']['crop_size']),
        length=config['dataset']['test_length'],
        device=device.type,
        split='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 1. Evaluate Pretrained Baseline
    print("\n--- Evaluating Pretrained Baseline ---")
    pretrained_matcher = DeepMatcher(max_keypoints=1024)
    pretrained_matcher.extractor = pretrained_matcher.extractor.to(device)
    pretrained_matcher.matcher = pretrained_matcher.matcher.to(device)
    
    pre_metrics = evaluate_model(pretrained_matcher, test_loader, device)
    
    # 2. Evaluate Fine-Tuned Model
    print("\n--- Evaluating Fine-Tuned Model ---")
    finetuned_matcher = DeepMatcher(max_keypoints=1024)
    finetuned_matcher.extractor = finetuned_matcher.extractor.to(device)
    finetuned_matcher.matcher = finetuned_matcher.matcher.to(device)
    
    # Load weights
    ckpt_path = Path(config['checkpointing']['save_dir']) / f"{config['checkpointing']['model_name']}_best.pth"
    if ckpt_path.exists():
        finetuned_matcher.matcher.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        ft_metrics = evaluate_model(finetuned_matcher, test_loader, device)
    else:
        print(f"Checkpoint {ckpt_path} not found! Run fine-tuning first.")
        ft_metrics = {"precision": 0, "recall": 0, "rmse": 0, "avg_inliers": 0}
        
    print("\n================================================")
    print("COMPARISON (Synthetic Test Set)")
    print(f"Metric\t\tPretrained\tFine-Tuned")
    print(f"Avg Inliers\t{pre_metrics['avg_inliers']:.2f}\t\t{ft_metrics['avg_inliers']:.2f}")
    print(f"Precision\t{pre_metrics['precision']:.4f}\t\t{ft_metrics['precision']:.4f}")
    print(f"Recall\t\t{pre_metrics['recall']:.4f}\t\t{ft_metrics['recall']:.4f}")
    print(f"RMSE (pixels)\t{pre_metrics['rmse']:.4f}\t\t{ft_metrics['rmse']:.4f}")
    print("================================================")

if __name__ == "__main__":
    main()
