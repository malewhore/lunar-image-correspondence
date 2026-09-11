import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Add root to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.lunar_dataset import LunarSyntheticDataset
from src.matching.matcher import DeepMatcher
from src.training.loss import GeometricMatchLoss

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("================================================")
    print("Phase 11: Fine-Tuning SuperPoint + LightGlue")
    print("================================================")
    
    config_path = "configs/finetune.yaml"
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Failed to load config {config_path}: {e}")
        return
        
    device_str = config['training']['device']
    if device_str == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
        
    print(f"Using device: {device}")
    
    # 1. Datasets & DataLoaders
    data_dir = config['dataset']['data_dir']
    crop_size = tuple(config['dataset']['crop_size'])
    
    train_dataset = LunarSyntheticDataset(
        data_dir=data_dir, crop_size=crop_size, 
        length=config['dataset']['train_length'], device=device.type, split='train'
    )
    val_dataset = LunarSyntheticDataset(
        data_dir=data_dir, crop_size=crop_size, 
        length=config['dataset']['val_length'], device=device.type, split='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # 2. Models
    matcher = DeepMatcher(max_keypoints=1024)
    matcher.extractor = matcher.extractor.to(device)
    matcher.matcher = matcher.matcher.to(device)
    
    # Freeze SuperPoint
    for param in matcher.extractor.parameters():
        param.requires_grad = False
        
    # Train LightGlue
    for param in matcher.matcher.parameters():
        param.requires_grad = True
        
    # 3. Optimizer & Loss
    optimizer = optim.AdamW(matcher.matcher.parameters(), lr=float(config['training']['learning_rate']))
    criterion = GeometricMatchLoss(match_threshold=config['training']['match_threshold'])
    
    # 4. Training Loop
    epochs = config['training']['epochs']
    best_val_loss = float('inf')
    
    ckpt_dir = Path(config['checkpointing']['save_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting Training...")
    
    for epoch in range(epochs):
        # TRAIN
        matcher.matcher.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            image0 = batch["image0"].to(device)
            image1 = batch["image1"].to(device)
            H_gt = batch["homography"].squeeze(0).to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                feats0 = matcher.extractor({"image": image0})
                feats1 = matcher.extractor({"image": image1})
            
            if feats0["keypoints"].shape[1] == 0 or feats1["keypoints"].shape[1] == 0:
                continue
            
            pred = {"image0": feats0, "image1": feats1}
            match_out = matcher.matcher(pred)
            
            matching_scores0 = match_out["matching_scores0"].squeeze(0)
            kpts0 = feats0["keypoints"].squeeze(0)
            kpts1 = feats1["keypoints"].squeeze(0)
            
            loss = criterion(kpts0, kpts1, matching_scores0, H_gt)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if (step + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        avg_train_loss = train_loss / len(train_loader)
        
        # VALIDATE
        matcher.matcher.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                image0 = batch["image0"].to(device)
                image1 = batch["image1"].to(device)
                H_gt = batch["homography"].squeeze(0).to(device)
                
                feats0 = matcher.extractor({"image": image0})
                feats1 = matcher.extractor({"image": image1})
                
                if feats0["keypoints"].shape[1] == 0 or feats1["keypoints"].shape[1] == 0:
                    continue
                
                pred = {"image0": feats0, "image1": feats1}
                match_out = matcher.matcher(pred)
                
                matching_scores0 = match_out["matching_scores0"].squeeze(0)
                kpts0 = feats0["keypoints"].squeeze(0)
                kpts1 = feats1["keypoints"].squeeze(0)
                
                loss = criterion(kpts0, kpts1, matching_scores0, H_gt)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"=== Epoch {epoch+1} Summary ===")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save Best Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = ckpt_dir / f"{config['checkpointing']['model_name']}_best.pth"
            torch.save(matcher.matcher.state_dict(), str(ckpt_path))
            print(f"[*] Saved new best checkpoint to {ckpt_path}")
            
    print("\nPhase 11 Fine-Tuning Complete!")
    print("================================================")

if __name__ == "__main__":
    main()
