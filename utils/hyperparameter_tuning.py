import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
import numpy as np
from decouple import config
from data.dataset import CDANDataset
from models.cdan import CDAN
import optuna
# ---------------------- Edge-Based Contrast (EBCM) and Sobel Loss ----------------------
def gradient_map(img):
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=img.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=img.device).view(1,1,3,3)
    g_x = F.conv2d(img, sobel_x, padding=1)
    g_y = F.conv2d(img, sobel_y, padding=1)
    return torch.sqrt(g_x**2 + g_y**2)
def ebcm(pred, target):
    pred_gray = pred.mean(1, keepdim=True)
    target_gray = target.mean(1, keepdim=True)
    grad_pred = gradient_map(pred_gray)
    grad_target = gradient_map(target_gray)
    return (grad_pred * grad_target).mean()
def edge_loss(pred, target):
    pred_gray = pred.mean(1, keepdim=True)
    target_gray = target.mean(1, keepdim=True)
    grad_pred = gradient_map(pred_gray)
    grad_target = gradient_map(target_gray)
    return F.mse_loss(grad_pred, grad_target)
# ---------------------- Objective Function ----------------------
def objective(trial):
    # Hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    epochs = trial.suggest_int('epochs', 50, 180, 5)
    lr = trial.suggest_float('lr', 1e-5, 1e-2)
    perceptual_loss_weight = trial.suggest_float('perceptual_loss_weight', 0.15 , 1)
    edge_loss_weight = trial.suggest_float('edge_loss_weight', 0.1, 1.0)
    vgg_layers = trial.suggest_categorical('vgg_layers', [16, 18, 20, 23, 25])
    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    # Dataset and DataLoader
    train_dataset = CDANDataset(os.path.join(DATASET_DIR_ROOT, 'train', 'low'),
                                os.path.join(DATASET_DIR_ROOT, 'train', 'high'),
                                train_transforms, train_transforms)
    test_dataset = CDANDataset(os.path.join(DATASET_DIR_ROOT, 'test', 'low'),
                               os.path.join(DATASET_DIR_ROOT, 'test', 'high'),
                               test_transforms, test_transforms)
    data_loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'validation': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }
    # Device
    device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    # Model, loss, optimizer
    model = CDAN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:vgg_layers].to(device)
    # Metrics
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity().to(device)
    # Track best metrics
    best_psnr = 0.0
    best_ssim = 0.0
    best_lpips = 0.0
    best_ebcm = 0.0

    for epoch in range(epochs):
        # ---------- Training ----------
        model.train()
        train_loss = 0.0
        train_psnr = 0.0
        train_ssim = 0.0
        train_lpips = 0.0
        train_ebcm = 0.0

        for inputs, targets in tqdm(data_loaders['train'], desc=f'Training Epoch {epoch+1}/{epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Combined loss
            mse_loss = criterion(outputs, targets)
            perceptual_loss_val = perceptual_loss_weight * F.mse_loss(vgg(outputs), vgg(targets))
            sobel_loss_val = edge_loss_weight * edge_loss(outputs, targets)
            loss = mse_loss + perceptual_loss_val + sobel_loss_val

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_psnr += psnr_metric(outputs, targets)
            train_ssim += ssim_metric(outputs, targets)
            train_lpips += lpips_metric(outputs, targets)
            train_ebcm += ebcm(outputs, targets)

        train_loss /= len(data_loaders['train'])
        train_psnr /= len(data_loaders['train'])
        train_ssim /= len(data_loaders['train'])
        train_lpips /= len(data_loaders['train'])
        train_ebcm /= len(data_loaders['train'])

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        val_lpips = 0.0
        val_ebcm = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(data_loaders['validation'], desc=f'Validation Epoch {epoch+1}/{epochs}'):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                mse_loss = criterion(outputs, targets)
                perceptual_loss_val = perceptual_loss_weight * F.mse_loss(vgg(outputs), vgg(targets))
                sobel_loss_val = edge_loss_weight * edge_loss(outputs, targets)
                loss = mse_loss + perceptual_loss_val + sobel_loss_val

                val_loss += loss.item()
                val_psnr += psnr_metric(outputs, targets)
                val_ssim += ssim_metric(outputs, targets)
                val_lpips += lpips_metric(outputs, targets)
                val_ebcm += ebcm(outputs, targets)

        val_loss /= len(data_loaders['validation'])
        val_psnr /= len(data_loaders['validation'])
        val_ssim /= len(data_loaders['validation'])
        val_lpips /= len(data_loaders['validation'])
        val_ebcm /= len(data_loaders['validation'])

        # Track best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_lpips = val_lpips
            best_ebcm = val_ebcm

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f}, PSNR: {train_psnr:.4f}, SSIM: {train_ssim:.4f}, LPIPS: {train_lpips:.4f}, EBCM: {train_ebcm:.6f}")
        print(f"Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}, LPIPS: {val_lpips:.4f}, EBCM: {val_ebcm:.6f}")
        print('-'*50)

    # Save best metrics in trial attributes
    trial.set_user_attr("best_metrics", {
        "PSNR": float(best_psnr),
        "SSIM": float(best_ssim),
        "LPIPS": float(best_lpips),
        "EBCM": float(best_ebcm)
    })

    return best_psnr

# ---------------------- Main ----------------------
if __name__ == '__main__':
    # Reproducibility
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    INPUT_SIZE = 224
    DATASET_DIR_ROOT = config('DATASET_DIR_ROOT')

    # Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_trial = study.best_trial
    best_metrics = best_trial.user_attrs["best_metrics"]

    print('Best hyperparameters found:')
    for param, value in best_params.items():
        print(f'{param}: {value}')

    print("Best Metrics:")
    print(f"PSNR: {best_metrics['PSNR']:.4f}")
    print(f"SSIM: {best_metrics['SSIM']:.4f}")
    print(f"LPIPS: {best_metrics['LPIPS']:.4f}")
    print(f"EBCM: {best_metrics['EBCM']:.6f}")
