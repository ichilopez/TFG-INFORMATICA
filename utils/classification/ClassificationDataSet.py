from torch.utils.data import Dataset
from PIL import Image

class ClassificationDataSet(Dataset):
    def __init__(self, input_paths, targets, transform=None):
        assert len(input_paths) == len(targets), "Input and target lists must be same length"
        self.input_paths = input_paths
        self.targets = targets
        self.transform_input = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_img = Image.open(self.input_paths[idx]).convert("L")  # o "RGB"
        if self.transform_input:
            input_img = self.transform_input(input_img)
        target_vector = self.targets[idx]

        return input_img, target_vector
