import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
class PairedDataset(Dataset):
    """
    Loads paired low-light and normal-light images.
    Expects two directories with the same number of files.
    Returns a dictionary with 'low' and 'normal' tensors.
    """
    def __init__(self, low_light_root, normal_light_root, image_size=[224, 224]):
        super().__init__()
        self.low_images = sorted(os.listdir(low_light_root))
        self.normal_images = sorted(os.listdir(normal_light_root))
        assert len(self.low_images) == len(self.normal_images), \
            "Low-light and normal-light datasets must have the same length"
        self.low_light_root = low_light_root
        self.normal_light_root = normal_light_root
        self.transforms = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_light_root, self.low_images[idx])
        normal_path = os.path.join(self.normal_light_root, self.normal_images[idx])

        low_img = Image.open(low_path).convert("RGB")
        normal_img = Image.open(normal_path).convert("RGB")

        low_tensor = self.transforms(low_img)
        normal_tensor = self.transforms(normal_img)

        return {"low": low_tensor, "normal": normal_tensor}
class UnpairedDataset(Dataset):
    """
    Loads only low-light images (e.g. for inference/testing).
    Returns tensor + filename.
    """
    def __init__(self, low_light_root, image_size=[224, 224]):
        super().__init__()
        self.low_light_dataset = [os.path.join(low_light_root, image)
                                  for image in os.listdir(low_light_root)]
        self.transforms = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.low_light_dataset)
    def __getitem__(self, idx):
        image_path = self.low_light_dataset[idx]
        low_light = Image.open(image_path).convert("RGB")
        low_light = self.transforms(low_light)
        filename = os.path.basename(image_path)
        return low_light, filename
