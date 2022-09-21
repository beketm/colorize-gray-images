import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as T

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        mode="train",
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        mode="validation",
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda", save_ith_image=4
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        if idx % save_ith_image == 0:
            x = x.to(device=device)
            with torch.no_grad():
                preds = model(x)
                fig, ax = plt.subplots( nrows=1, ncols=3 )
                ax[0].set_axis_off()
                ax[1].set_axis_off()
                ax[2].set_axis_off()

                ax[0].imshow(T.ToPILImage()(x[0]), cmap='gray')
                ax[1].imshow(T.ToPILImage()(preds[0]))
                ax[2].imshow(T.ToPILImage()(y[0]))
                fig.savefig(f"{folder}{idx}.png",  bbox_inches='tight')
                plt.close()
    model.train()

