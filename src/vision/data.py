from __future__ import annotations
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO

def get_pneumoniamnist_loaders(batch_size: int = 64, img_size: int = 224, num_workers: int = 2):
    """PneumoniaMNIST: chest X-ray images for pneumonia detection (binary classification)."""
    info = INFO["pneumoniamnist"]
    DataClass = getattr(medmnist, info["python_class"])

    tfm_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1ch -> 3ch
        # light augmentation for regularization:
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
    ])
    tfm_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])

    train_set = DataClass(split="train", transform=tfm_train, download=True)
    val_set = DataClass(split="val", transform=tfm_eval, download=True)
    test_set = DataClass(split="test", transform=tfm_eval, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


    return train_loader, val_loader, test_loader, info
