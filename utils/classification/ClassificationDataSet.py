from torch.utils.data import Dataset
from PIL import Image

class ClassificationDataSet(Dataset):
    def __init__(self, data_info, transform=None):

        self.data_info = data_info
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]

        img = Image.open(info['file_path_image']).convert("L")  # escala de grises

        if self.transform:
            img = self.transform(img)

        label = info['label']

        return img, label
