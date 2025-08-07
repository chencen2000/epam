from typing import Tuple, Dict
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

def get_transforms(target_size: Tuple[int, int] = None, config:Dict=None) -> A.Compose:
        """Get preprocessing transforms"""
        if target_size is None:
            target_size = config.get('target_size', (1792, 1792))
            
        return A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2()
        ])
