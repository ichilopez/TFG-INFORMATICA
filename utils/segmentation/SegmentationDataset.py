import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import lightning as L
from torchvision.transforms import InterpolationMode

class SegmentationDataset(Dataset):
    def __init__(self, images_path, image_size=(256, 256)):
        self.images_path = images_path
        self.image_size = image_size

        self.img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        self.image_files = os.listdir(images_path)

    def __len__(self):
     return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]

        img_path = os.path.join(self.images_path, img_file, "0.jpeg")
        mask_path = os.path.join(self.images_path, img_file, "2.jpeg")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.img_transform(img)
        mask = self.mask_transform(mask)

        return img, mask

