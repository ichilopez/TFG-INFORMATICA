from torch.utils.data import Dataset
from PIL import Image

class ClassificationDataSet(Dataset):
    def __init__(self, data_info, transform=None):
        self.data_info = data_info
        self.transform = transform
        self.label_map = {"BENIGN": 0, "MALIGNANT": 1}

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]
        img_path = info['file_path_image']

        img = Image.open(img_path).convert("L")  

        if self.transform:
            img = self.transform(img)

        label_str = str(info['label']).strip().upper()
        label = self.label_map.get(label_str, -1)

        return img, label, img_path
