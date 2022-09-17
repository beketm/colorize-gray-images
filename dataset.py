
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision.transforms as T

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform is not None:
            image = self.transform(image)
            mask =  self.transform(mask)

        return image, mask
    
def test():
    ds = CarvanaDataset(
        image_dir="images/gray",
        mask_dir="images/color",
        transform=T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
            T.CenterCrop(208),
            T.Normalize([0.0], [1.0])
        ])
    )

    print(ds.__getitem__(5)[0].shape)
    print(ds.__getitem__(5)[1].shape)


if __name__=="__main__":
    test()