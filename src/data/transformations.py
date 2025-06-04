from typing import Tuple
import albumentations as A


def get_train_transforms(target_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
        A.pytorch.ToTensorV2()
    ])


def get_val_transforms(target_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.pytorch.ToTensorV2()
    ])
