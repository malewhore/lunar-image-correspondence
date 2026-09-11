import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Add root to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.lunar_dataset import LunarSyntheticDataset
from src.matching.matcher import DeepMatcher
from src.training.loss import GeometricMatchLoss

def main():
    print("================================================")
    print("Phase 10: Fine-Tuning SuperPoint + LightGlue")
    print("================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset & DataLoader
    data_dir = "data/real_pairs_drive"
    dataset = LunarSyntheticDataset(data_dir=data_dir, crop_size=(512, 512), length=100, device=device.type)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 2. Models
    # We use our DeepMatcher which wraps SuperPoint and LightGlue
    matcher = DeepMatcher(max_keypoints=1024)
    matcher.extractor = matcher.extractor.to(device)
    matcher.matcher = matcher.matcher.to(device)
    
    # We freeze SuperPoint (feature extractor) to save memory and focus on the matching head
    for param in matcher.extractor.parameters():
        param.requires_grad = False
        
    # We train LightGlue
    for param in matcher.matcher.parameters():
        param.requires_grad = True
        
    matcher.matcher.train()
    
    # 3. Optimizer & Loss
    optimizer = optim.AdamW(matcher.matcher.parameters(), lr=1e-4)
    criterion = GeometricMatchLoss(match_threshold=5.0)
    
    # 4. Training Loop (Dry-run for 5 steps)
    print("\nStarting Training (5 steps dry-run)...")
    
    for step, batch in enumerate(dataloader):
        if step >= 5:
            break
            
        # Move to device
        image0 = batch["image0"].to(device) # [B, 1, H, W]
        image1 = batch["image1"].to(device)
        H_gt = batch["homography"].squeeze(0).to(device) # [3, 3]
        
        optimizer.zero_grad()
        
        # --- Forward Pass ---
        # 1. Extract features (frozen)
        with torch.no_grad():
            feats0 = matcher.extractor({"image": image0})
            feats1 = matcher.extractor({"image": image1})
        
        # 2. Match features (trainable)
        # We need to construct the input dict for lightglue
        pred = {"image0": feats0, "image1": feats1}
        match_out = matcher.matcher(pred)
        
        matching_scores0 = match_out["matching_scores0"].squeeze(0) # [N]
        kpts0 = feats0["keypoints"].squeeze(0) # [N, 2]
        kpts1 = feats1["keypoints"].squeeze(0) # [M, 2]
        
        # --- Loss Calculation ---
        loss = criterion(kpts0, kpts1, matching_scores0, H_gt)
        
        # --- Backward Pass ---
        loss.backward()
        optimizer.step()
        
        print(f"Step [{step+1}/5] - Geometric Match Loss: {loss.item():.4f}")

    print("\nTraining Dry-Run Complete! Forward and Backward pass successful.")
    
    # Save a checkpoint
    ckpt_dir = Path("outputs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "lightglue_lunar_step5.pth"
    torch.save(matcher.matcher.state_dict(), str(ckpt_path))
    print(f"Saved dry-run checkpoint to: {ckpt_path}")
    print("================================================")

if __name__ == "__main__":
    main()
