import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
# === Dataset ===
class CVCCColonDBDataset(Dataset):
    def __init__(self, enhanced_dir, high_dir, transform=None):
        self.enhanced_dir = enhanced_dir
        self.high_dir = high_dir
        self.transform = transform
        self.image_names = sorted([
            f for f in os.listdir(enhanced_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and
               os.path.exists(os.path.join(high_dir, f))
        ])
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, idx):
        enhanced_path = os.path.join(self.enhanced_dir, self.image_names[idx])
        high_path = os.path.join(self.high_dir, self.image_names[idx])
        enhanced_img = Image.open(enhanced_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")
        if self.transform:
            enhanced_img = self.transform(enhanced_img)
            high_img = self.transform(high_img)
        return enhanced_img, high_img
# === Define Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# === Paths (example usage) ===
train_enhanced_dir = "/content/drive/MyDrive/Colon_Enhanced/train_enhanced"
train_high_dir = "/content/cvccolondbsplit/train/high"
val_enhanced_dir = "/content/drive/MyDrive/Colon_Enhanced/val_enhanced"
val_high_dir = "/content/cvccolondbsplit/val/high"
# === Create Dataset Instances ===
train_dataset = CVCCColonDBDataset(train_enhanced_dir, train_high_dir, transform=transform)
val_dataset = CVCCColonDBDataset(val_enhanced_dir, val_high_dir, transform=transform)
# === Create DataLoaders ===
batch_size = 8
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
