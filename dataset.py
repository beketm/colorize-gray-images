
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision.transforms as T
from datasets import load_dataset, Image


class CarvanaDataset(Dataset):
    def __init__(self, mode="train", transform=None):
        self.dataset = load_dataset("frgfm/imagenette", '160px', split=mode)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image = self.dataset[index]['image'].convert("L")
        mask = self.dataset[index]['image'].convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
            mask =  self.transform(mask)

        return image, mask
    
def test():
    ds = CarvanaDataset(
        mode="validation",
        transform=T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
            T.CenterCrop(160),
            T.Normalize([0.0], [1.0])
        ])
    )

    print(ds.__getitem__(5)[1].shape)
    print(ds.__getitem__(5)[0].shape)



if __name__=="__main__":
    test()