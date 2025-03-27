import random
import shutil
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from dataset import CarvanaDataset
from utils.metrics import Accuracy, DiceScore, IoU
from utils.visualization import MetricsVisualizer


def save_checkpoint(state, filename='output/checkpoints/my_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    folder = Path(filename)
    folder.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])


def split_train_val(train_dir, train_maskdir, val_dir, val_maskdir, split_ratio=0.2, revert=False):
    train_dir = Path(train_dir)
    train_maskdir = Path(train_maskdir)
    val_dir = Path(val_dir)
    val_maskdir = Path(val_maskdir)

    if revert:
        for val, val_mask in zip(val_dir.glob('*.jpg'), val_maskdir.glob('*.gif')):
            shutil.move(val, train_dir / val.name)
            shutil.move(val_mask, train_maskdir / val_mask.name)

    train_images = sorted(train_dir.glob('*.jpg'))
    train_masks = sorted(train_maskdir.glob('*.gif'))

    num_val_samples = int(len(train_images) * split_ratio)
    indices = random.sample(range(len(train_images)), num_val_samples)
    # num_train_samples = 4600
    # indices = list(range(num_train_samples, len(train_images)))

    for idx in indices:
        shutil.move(train_images[idx], val_dir / train_images[idx].name)
        shutil.move(train_masks[idx], val_maskdir / train_masks[idx].name)

    print(f'Split {len(indices)}/{len(train_images)} images and masks for validation')


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0  # dice score is a metric that measures the similarity between two sets of data
    model.eval()
    acc_calculator = Accuracy(data_format='BCHW')
    dice_calculator = DiceScore(smooth=1e-6)
    iou_calculator = IoU(smooth=1e-6)
    metrics_visualizer = MetricsVisualizer(title="Metrics Scores", figsize=(10, 5))

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            pred = model(x)
            preds = torch.sigmoid(pred)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            acc_calculator.update(pred, y)
            dice_calculator.update(pred, y)
            iou_calculator.update(pred, y)

    print(
        f'Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}'
    )
    print(f'New Accuracy: {acc_calculator.compute():.4f}')
    print(f'Old Dice score: {dice_score / len(loader):.4f}')
    print(f'New Dice score: {dice_calculator.compute():.4f}')
    print(f'IoU score: {iou_calculator.compute():.4f}')
    metrics_visualizer.plot(
        dice_calculator.metric_history,
        save_path='output/metrics/dice_score.png',
        show=False,
        close=True)
    metrics_visualizer.plot(
        iou_calculator.metric_history,
        save_path='output/metrics/iou_score.png',
        show=False,
        close=True)
    metrics_visualizer.plot(
        acc_calculator.metric_history,
        save_path='output/metrics/accuracy.png',
        show=False,
        close=True)
    model.train()


def save_predictions_as_imgs(
        loader, model, folder='output/saved_images', device='cuda'
):
    model.eval()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, folder / f'pred_{idx}.png'
        )
        torchvision.utils.save_image(
            y.unsqueeze(1), folder / f'truth_{idx}.png'
        )
    model.train()
