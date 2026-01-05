import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(strength=1.0):
    """strength: 0.0(약함) ~ 1.0(강함)"""
    return A.Compose([
        # A.Resize(224, 224),
                A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(min_scale, 1.0),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5 * strength),
        A.CoarseDropout(p=0.5 * strength),
        A.HorizontalFlip(p=0.5 * strength),
        A.Rotate(limit=int(15 * strength), p=0.5 * strength),
        A.RandomBrightnessContrast(p=0.3 * strength),
        A.GaussNoise(var_limit=(1.0, 30.0 * strength), p=0.3 * strength),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
