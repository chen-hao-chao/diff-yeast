import os
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff

class CellDataLoader(Dataset):
    def __init__(self, root_dir, root_dir_no_gfp, data_list):
        self.data_list = data_list
        self.root_dir_no_gfp = root_dir_no_gfp
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image = tiff.imread(os.path.join(self.root_dir, self.data_list[index]))
        image_no_gfp = 0 #tiff.imread(os.path.join(self.root_dir_no_gfp, self.data_list[index]))
        return image, image_no_gfp