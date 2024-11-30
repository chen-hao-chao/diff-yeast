import os
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff

class MyData(Dataset):
    def __init__(self, root_dir, data_list, phase):
        self.data_list = data_list
        self.phase = phase
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image = tiff.imread(os.path.join(self.root_dir, self.data_list[index]))
        label = self.phase
        return image, label