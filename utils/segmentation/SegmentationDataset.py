import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import lightning as L

class SegmentationDataset(Dataset):
    def __init__(self, images_path, image_size=(256, 256)):
        self.images_path = images_path
        self.image_size = image_size
        self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),  # convierte a tensor [0,1]
            ])

        self.image_files = os.listdir(images_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
     img_file = self.image_files[idx]

     img_path = os.path.join(self.images_path, img_file, "0.jpeg")
     mask_path = os.path.join(self.images_path, img_file, "2.jpeg")  

     img = Image.open(img_path).convert("RGB")  # 3 canales
     mask = Image.open(mask_path).convert("L")  # 1 canal

     img = self.transform(img)

     mask = transforms.ToTensor()(mask)  # shape: [1, H, W]
     return img, mask

