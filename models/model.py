import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.models as models

from tqdm import tqdm
from PIL import Image
import numpy as np

from models.base import BaseModel
from utils.post_processing import enhance_color, enhance_contrast


class Model(BaseModel):
    def __init__(self, network, **kwargs):
        """Initialize BaseModel with kwargs."""
        super(Model, self).__init__(**kwargs)

        self.network = network.to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    # ---------------------------
    # Edge Loss (Sobel operator)
    # ---------------------------
    def edge_loss(self, outputs, targets):
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        def sobel_filter(img):
            img_gray = img.mean(dim=1, keepdim=True)  # grayscale
            grad_x = F.conv2d(img_gray, sobel_x, padding=1)
            grad_y = F.conv2d(img_gray, sobel_y, padding=1)
            return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        edge_out = sobel_filter(outputs)
        edge_tgt = sobel_filter(targets)
        return F.l1_loss(edge_out, edge_tgt)

    # ---------------------------
    # Composite Loss
    # ---------------------------
    def composite_loss(self, outputs, targets):
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:20].to(self.device)

        mse_loss = self.criterion(outputs, targets)
        perceptual_loss = 0.25 * F.mse_loss(vgg(outputs), vgg(targets))
        edge_loss_val = 0.1 * self.edge_loss(outputs, targets)

        return mse_loss + perceptual_loss + edge_loss_val

    # ---------------------------
    # Save Output Images
    # ---------------------------
    def generate_output_images(self, outputs, filenames, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for i, output_image in enumerate(outputs):
            output_image = output_image.detach().cpu().permute(1, 2, 0).numpy()
            output_image = (output_image * 255).clip(0, 255).astype(np.uint8)
            output_image = Image.fromarray(output_image)
            output_path = os.path.join(save_dir, filenames[i])
            output_image.save(output_path)
        print(f'{len(outputs)} output images generated and saved to {save_dir}')

    # ---------------------------
    # Training Step
    # ---------------------------
    def train_step(self):
        train_losses = np.zeros(self.epoch)
        best_loss = float('inf')
        self.network.to(self.device)

        for epoch in range(self.epoch):
            train_loss = 0.0
            dataloader_iter = tqdm(self.dataloader, desc=f'Training... Epoch: {epoch + 1}/{self.epoch}', total=len(self.dataloader))
            for inputs, targets in dataloader_iter:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.network(inputs)
                loss = self.composite_loss(outputs, targets)
                train_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                dataloader_iter.set_postfix({'loss': loss.item()})

            train_loss = train_loss / len(self.dataloader)

            if train_loss < best_loss:
                best_loss = train_loss
                self.save_model(self.network)

            train_losses[epoch] = train_loss
            print(f"Epoch [{epoch + 1}/{self.epoch}] Train Loss: {train_loss:.4f}")

    # ---------------------------
    # Compute EBCM (Edge-Based Contrast Measure)
    # ---------------------------
    def compute_ebcm(self, image_tensor):
        image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
        image = (image * 255).clip(0, 255).astype(np.uint8)
        gray = np.array(Image.fromarray(image).convert("L"), dtype=np.float32)
        gx = np.abs(np.gradient(gray, axis=0))
        gy = np.abs(np.gradient(gray, axis=1))
        edge_strength = np.sqrt(gx ** 2 + gy ** 2)
        return edge_strength.mean()

    # ---------------------------
    # Testing Step
    # ---------------------------
    def test_step(self):
        path = os.path.join(self.model_path, self.model_name)
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.eval()

        psnr = PeakSignalNoiseRatio().to(self.device)
        ssim = StructuralSimilarityIndexMeasure().to(self.device)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.device)

        all_outputs = []
        all_filenames = []
        ebcm_scores = []

        with torch.no_grad():
            test_loss, test_psnr, test_ssim, test_lpips = 0.0, 0.0, 0.0, 0.0
            self.network.eval()
            self.optimizer.zero_grad()

            if self.is_dataset_paired:
                for inputs, targets in tqdm(self.dataloader, desc='Testing...'):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.network(inputs)
                    if self.apply_post_processing:
                        outputs = enhance_contrast(outputs, contrast_factor=1.12)
                        outputs = enhance_color(outputs, saturation_factor=1.35)
                    loss = self.criterion(outputs, targets)
                    test_loss += loss.item()
                    test_psnr += psnr(outputs, targets)
                    test_ssim += ssim(outputs, targets)
                    test_lpips += lpips(outputs, targets)

                    # EBCM metric
                    for img in outputs:
                        ebcm_scores.append(self.compute_ebcm(img))

                    all_outputs.append(outputs)
            else:
                for inputs, filenames in tqdm(self.dataloader, desc='Testing...'):
                    inputs = inputs.to(self.device)
                    outputs = self.network(inputs)

                    # EBCM metric
                    for img in outputs:
                        ebcm_scores.append(self.compute_ebcm(img))

                    all_outputs.append(outputs)
                    all_filenames.extend(filenames)

            if self.is_dataset_paired:
                test_loss /= len(self.dataloader)
                test_psnr /= len(self.dataloader)
                test_ssim /= len(self.dataloader)
                test_lpips /= len(self.dataloader)
                avg_ebcm = np.mean(ebcm_scores)
                print(f'Test Loss: {test_loss:.4f}, '
                      f'Test PSNR: {test_psnr:.4f}, '
                      f'Test SSIM: {test_ssim:.4f}, '
                      f'Test LPIPS: {test_lpips:.4f}, '
                      f'Test EBCM: {avg_ebcm:.4f}')

            # Concatenate and save results
            all_outputs = torch.cat(all_outputs, dim=0)
            self.generate_output_images(all_outputs, all_filenames, self.output_images_path)
