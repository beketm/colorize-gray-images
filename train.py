import torch
from tqdm import tqdm
import torchvision.transforms as T

import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import gc

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 160  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "images/gray"
TRAIN_MASK_DIR = "images/color"
VAL_IMG_DIR = "images/gray"
VAL_MASK_DIR = "images/color"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        torch.cuda.empty_cache()

        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)
        # print(data.shape)
        # print(targets.shape)
        
        # forward
        predictions = model(data)
        assert predictions.shape == targets.shape
        loss = loss_fn(predictions, targets)
        print(predictions.shape)

        with torch.no_grad():
            optimizer.zero_grad()
            # backward
            loss.backward()
            optimizer.step()
            

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    torch.cuda.empty_cache()
    train_transform = T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
            T.CenterCrop(IMAGE_HEIGHT),
            T.Normalize([0.0], [1.0])
        ])

    val_transforms = T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
            T.CenterCrop(IMAGE_HEIGHT),
            T.Normalize([0.0], [1.0]),
        ])

    model = UNET().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for param in model.parameters():
        param.requires_grad = True

    train_loader, val_loader = get_loaders(
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    # check_accuracy(val_loader, model, device=DEVICE)
    # scaler = torch.cuda.amp.GradScaler()
    scaler = None

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        # save_checkpoint(checkpoint)

        # check accuracy
        # check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
    
    save_predictions_as_imgs(
        val_loader, model, folder="saved_images/", device=DEVICE
    )


if __name__ == "__main__":
    main()