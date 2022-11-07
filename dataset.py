
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision.datasets import CIFAR100
import torchvision.transforms as T
from datasets import load_dataset, Image
from skimage import io, color



class CarvanaDataset(Dataset):
    def __init__(self):
        self.dataset = CIFAR100(root="./dataset", download=True)

    def __len__(self):
        return 15
        # return len(self.dataset)

    def __getitem__(self, index):
        
        rgb = io.imread(self.dataset[index][0])
        lab = color.rgb2lab(rgb)
        lab = np.moveaxis(lab, 2, 0)

        gray_scaled_image = torch.from_numpy(lab[0])
        ab_values = torch.from_numpy(lab[1:3])

        return gray_scaled_image, ab_values
    
def test():
    ds = CarvanaDataset()

    print(ds.__getitem__(5)[1].shape)
    print(ds.__getitem__(5)[0].shape)



if __name__=="__main__":
    test()