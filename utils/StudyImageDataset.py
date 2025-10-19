
from torch.utils.data import Dataset
from PIL import Image

class StudyImageDataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Cargar imagen de entrada
        img = Image.open(self.data_paths[idx]).convert("L")  # escala de grises
        if self.transform:
            img = self.transform(img)
        return img
