from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    """
    Carvana dataset for image segmentation.
    """
    def __init__(self, image_dir: Path, mask_dir: Path, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(self.image_dir.glob("*.jpg"))
        self.masks = sorted(self.mask_dir.glob("*.gif"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0  # used for the sigmoid activation function

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


def test():
    image_dir = Path("input/Carvana/train")
    mask_dir = Path("input/Carvana/train_masks")
    dataset = CarvanaDataset(image_dir, mask_dir)
    print(len(dataset))
    image, mask = dataset[0]
    print(image.shape)
    print(mask.shape)


if __name__ == "__main__":
    test()
