import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
def simulate_endoscopic_degradation(image_rgb, brightness_range=(0.6, 0.9),
                                    gaussian_blur_sigma=0.8, noise_level=0.005):
    """
    Simulates realistic low-light colonoscopy degradation.
    Works FULLY in RGB space to avoid BGR/RGB confusion.
    """
    degraded_image = image_rgb.astype(np.float32)
    height, width, _ = degraded_image.shape
    # 1. Mild Vignetting (less extreme)
    center_x, center_y = width // 2, height // 2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    normalized_dist = dist_from_center / max_dist
    exponent = random.uniform(0.5, 0.8)
    vignette_mask = 1 - (normalized_dist**exponent * 0.25)  # max 25% darker corners
    vignette_mask = np.stack([vignette_mask]*3, axis=-1)
    degraded_image *= vignette_mask
    # 2. Brightness scaling
    brightness_factor = random.uniform(*brightness_range)
    degraded_image *= brightness_factor
    # 3. Gaussian blur
    if gaussian_blur_sigma > 0:
        kernel_size = 3
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        degraded_image = cv2.GaussianBlur(degraded_image, (kernel_size, kernel_size),
                                          sigmaX=gaussian_blur_sigma)
    # 4. Add Poisson noise
    noise = np.random.poisson(degraded_image * noise_level).astype(np.float32)
    degraded_image += noise
    # 5. Slight color cast (reddish colon tones)
    red_scale   = random.uniform(1.05, 1.15)
    green_scale = random.uniform(0.95, 1.05)
    blue_scale  = random.uniform(0.85, 0.95)
    degraded_image[..., 0] *= red_scale   # R
    degraded_image[..., 1] *= green_scale # G
    degraded_image[..., 2] *= blue_scale  # B
    degraded_image = np.clip(degraded_image, 0, 255).astype(np.uint8)
    return degraded_image
def prepare_dataset(input_dir, output_dir, val_ratio=0.1, test_ratio=0.2):
    """
    Prepares a low-light dataset by splitting a directory of high-res images
    and applying a custom degradation function. Saves images in original size.
    """
    print(f"Input directory: {input_dir}")
    image_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_list.sort()
    if len(image_list) == 0:
        print("No images found in", input_dir)
        return
    # Split into train/val/test
    trainval_imgs, test_imgs = train_test_split(image_list, test_size=test_ratio, random_state=42)
    train_imgs, val_imgs = train_test_split(trainval_imgs, test_size=val_ratio / (1 - test_ratio), random_state=42)
    splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
    for split, filenames in splits.items():
        print(f"\nProcessing {split.upper()} split with {len(filenames)} images...")
        low_dir = os.path.join(output_dir, split, 'low')
        high_dir = os.path.join(output_dir, split, 'high')
        os.makedirs(low_dir, exist_ok=True)
        os.makedirs(high_dir, exist_ok=True)
        for fname in tqdm(filenames):
            image_path = os.path.join(input_dir, fname)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping unreadable image: {fname}")
                continue
            # Convert BGR → RGB and keep original size
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            degraded_img = simulate_endoscopic_degradation(rgb_image)
            # Save GT (high) and degraded (low)
            Image.fromarray(rgb_image).save(os.path.join(high_dir, fname))
            Image.fromarray(degraded_img).save(os.path.join(low_dir, fname))
    print("\n✅ Dataset preparation complete. Images saved in original size, RGB format.")
if __name__ == "__main__":
    original_dataset_path = "/content/cvccolondb/data/train/images"  # change to your source path
    output_dataset_path = "/content/cvccolondbsplit"                  # where to save train/val/test
    prepare_dataset(
        input_dir=original_dataset_path,
        output_dir=output_dataset_path,
        val_ratio=0.1,
        test_ratio=0.2
    )
