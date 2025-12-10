import os, shutil, random
from typing import Tuple, List
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE = 300
DEFAULT_CLASSES = ["acne", "dark_spot", "redness", "normal"]
IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".pgm",".tif",".tiff",".webp")

def _exists(p: str) -> bool:
    return os.path.isdir(p)

def _ensure_classes(root: str, classes: List[str]) -> None:
    miss = [c for c in classes if not _exists(os.path.join(root, c))]
    if miss:
        raise FileNotFoundError(
            f"Expected class folders in '{root}': {classes}\nMissing: {miss}"
        )

def _has_images(folder: str) -> bool:
    if not _exists(folder): return False
    for c in os.listdir(folder):
        cdir = os.path.join(folder, c)
        if os.path.isdir(cdir):
            if any(f.lower().endswith(IMG_EXTS) for f in os.listdir(cdir)):
                return True
    return False

def _auto_split_if_needed(data_dir: str, classes: List[str], val_ratio: float = 0.15):
    train_root = os.path.join(data_dir, "train")
    val_root   = os.path.join(data_dir, "val")
    if _exists(train_root) and _exists(val_root) and _has_images(train_root) and _has_images(val_root):
        return  # good

    # case: data/<class>/images...
    if all(_exists(os.path.join(data_dir, c)) for c in classes):
        os.makedirs(train_root, exist_ok=True)
        os.makedirs(val_root, exist_ok=True)
        for c in classes:
            src = os.path.join(data_dir, c)
            dst_tr = os.path.join(train_root, c); os.makedirs(dst_tr, exist_ok=True)
            dst_va = os.path.join(val_root,   c); os.makedirs(dst_va, exist_ok=True)
            imgs = [f for f in os.listdir(src) if f.lower().endswith(IMG_EXTS)]
            random.shuffle(imgs)
            k = max(1, int(len(imgs)*val_ratio)) if len(imgs)>1 else 1
            valset = set(imgs[:k])
            for f in imgs:
                srcf = os.path.join(src, f)
                dstd = dst_va if f in valset else dst_tr
                dstf = os.path.join(dstd, f)
                if not os.path.exists(dstf):
                    shutil.copy2(srcf, dstf)

def make_loaders(data_dir: str = "data", batch_size: int = 16,
                 classes: List[str] = DEFAULT_CLASSES) -> Tuple[DataLoader, DataLoader, List[str]]:
    _auto_split_if_needed(data_dir, classes)

    train_root = os.path.join(data_dir, "train")
    val_root   = os.path.join(data_dir, "val")

    if not (_exists(train_root) and _exists(val_root)):
        raise FileNotFoundError(
            "Couldn't find data folders.\nLooked for:\n"
            f"  {train_root}\n  {val_root}\n"
            "Fix: create data/train and data/val with class subfolders, or place data/<class> and I'll split."
        )
    _ensure_classes(train_root, classes)
    _ensure_classes(val_root, classes)

    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(train_root, transform=train_tfms)
    val_ds   = datasets.ImageFolder(val_root,   transform=val_tfms)
    print("Detected classes:", train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, train_ds.classes
