from pathlib import Path

import albumentations as A
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from models import UNet, NestedUNet, ResUNet, AttentionUNet, ResAttentionUNet
from utils.loss import FocalLoss
from utils.utils import (check_accuracy, get_loaders, load_checkpoint,
                         save_checkpoint, save_predictions_as_imgs,
                         split_train_val)
from utils.visualization import LossVisualizer

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = (
    'mps'
    if torch.backends.mps.is_available()
    else 'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)
BATCH_SIZE = 8
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 480
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = Path('input/Carvana/train/')
TRAIN_MASK_DIR = Path('input/Carvana/train_masks/')
VAL_IMG_DIR = Path('input/Carvana/val/')
VAL_MASK_DIR = Path('input/Carvana/val_masks/')
WORKING_DIR = Path('output/')


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        if DEVICE == 'mps':
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        else:
            with torch.amp.autocast(device_type=DEVICE):
                predictions = model(data)
                loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        if DEVICE == 'mps':
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean', multi_class=False, data_format='BCHW')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_visualizer = LossVisualizer(title="Training Loss", figsize=(10, 6))

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load('output/checkpoints/my_checkpoint.pth.tar'), model)
        check_accuracy(val_loader, model, device=DEVICE)

    scaler = torch.amp.GradScaler(device=DEVICE)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        loss_visualizer.plot(
            loss_fn.loss_history,
            save_path=WORKING_DIR / 'loss' / f'loss_curve_{epoch}.png',
            show=False,
            close=True)

        # save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder=WORKING_DIR / 'saved_images', device=DEVICE
        )


if __name__ == '__main__':
    # split_train_val(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, split_ratio=0.3, revert=True)
    main()
