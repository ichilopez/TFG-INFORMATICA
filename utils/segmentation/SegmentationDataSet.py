from torch.utils.data import Dataset
from PIL import Image

class SegmentationDataSet(Dataset):
    def __init__(self, input_paths, target_paths, transform=None):
        assert len(input_paths) == len(target_paths), "Input and target lists must be same length"
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform_input = transform
        self.transform_target = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_img = Image.open(self.input_paths[idx]).convert("L") 
        target_img = Image.open(self.target_paths[idx]).convert("L")

        if self.transform_input:
            input_img = self.transform_input(input_img)
        if self.transform_target:
            target_img = self.transform_target(target_img)

        return input_img, target_img
