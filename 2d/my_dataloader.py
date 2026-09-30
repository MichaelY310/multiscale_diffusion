import os
from typing import Tuple, Optional, List, Sequence

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def _mnist_transforms():
    """
    Important (Windows): avoid lambda/closures here, otherwise DataLoader(num_workers>0)
    can't pickle the transform and will crash with:
      "Can't get local object '<lambda>'"
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),  # [1,28,28] in [0,1]
            transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST normalization
        ]
    )


def _cifar10_transforms():
    """
    Returns RGB tensor in [-1, 1] range (mean=0.5,std=0.5).
    Also avoids lambda/closures for Windows multiprocessing compatibility.
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),  # [3,32,32] in [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1,1]
        ]
    )


def _anime_transforms(image_size: int = 256):
    """
    Basic image transforms for a local folder dataset.

    Output:
      - RGB tensor [3, image_size, image_size] in [-1, 1]

    Note (Windows): avoid Lambda/lambda for DataLoader multiprocessing compatibility.
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1,1]
        ]
    )


class AnimeImageDataset(Dataset):
    """
    Loads images from a local folder (e.g. ./anime/*.jpg).

    Returns:
      - image: torch.FloatTensor [3,H,W] in [-1,1]
      - path: str (file path)
    """

    _EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    def __init__(self, root_dir: str = "./anime", image_size: int = 256, recursive: bool = False):
        self.root_dir = root_dir
        self.image_size = image_size
        self.recursive = recursive

        # Build file list once to keep __getitem__ fast.
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Anime folder not found: {self.root_dir}")

        paths: List[str] = []
        if recursive:
            for dirpath, _, filenames in os.walk(self.root_dir):
                for fn in filenames:
                    if fn.lower().endswith(self._EXTS):
                        paths.append(os.path.join(dirpath, fn))
        else:
            for fn in os.listdir(self.root_dir):
                if fn.lower().endswith(self._EXTS):
                    paths.append(os.path.join(self.root_dir, fn))

        paths.sort()
        if len(paths) == 0:
            raise FileNotFoundError(f"No images found in: {self.root_dir}")

        self.paths = paths
        self.transform = _anime_transforms(image_size=image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image

        path = self.paths[idx]
        # Always convert to RGB to guarantee 3 channels.
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, path


class LSUNLmdbChurchOutdoorDataset(Dataset):
    """
    Loads images from an LSUN LMDB database (e.g. church_outdoor_train_lmdb).

    Images are stored as raw JPEG bytes; we decode them on-the-fly.
    Keys are pre-loaded into a list to support O(1) random access.

    Returns:
      - image: torch.FloatTensor [3, image_size, image_size] in [-1, 1]
      - key:   str (the LMDB key, for debugging)
    """

    def __init__(self, db_path: str, image_size: int = 256):
        import lmdb as _lmdb

        self.db_path = db_path
        self.image_size = image_size
        self.transform = _anime_transforms(image_size=image_size)

        env = _lmdb.open(db_path, readonly=True, lock=False,
                         readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.keys = [k for k, _ in txn.cursor()]
        env.close()

        self._env = None

    def _get_env(self):
        if self._env is None:
            import lmdb as _lmdb
            self._env = _lmdb.open(self.db_path, readonly=True, lock=False,
                                   readahead=False, meminit=False)
        return self._env

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        import io
        from PIL import Image

        env = self._get_env()
        with env.begin(write=False) as txn:
            val = txn.get(self.keys[idx])

        img = Image.open(io.BytesIO(val)).convert("RGB")
        x = self.transform(img)
        return x, self.keys[idx].decode("ascii")


def get_anime_dataloader(
    data_root: str = "./anime",
    *,
    batch_size: int = 8,
    image_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
    recursive: bool = False,
) -> DataLoader:
    """
    DataLoader for local anime image folder.

    Returns batches:
      - images: [B, 3, image_size, image_size] float32 in [-1,1]
      - paths: list[str] (length B)
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataset = AnimeImageDataset(
        root_dir=data_root, image_size=image_size, recursive=recursive
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )



def get_mnist_dataloader(
    data_root: str = "./data/MNIST",
    *,
    train: bool = True,
    batch_size: int = 8,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
) -> DataLoader:
    """
    MNIST handwritten digits DataLoader.

    Returns batches:
      - images: [B, 1, 28, 28] float32 (normalized)
      - labels: [B] int64
    """
    try:
        from torchvision.datasets import MNIST
    except Exception as e:
        raise ImportError(
            "torchvision is required for MNIST. Install it via pip: pip install torchvision"
        ) from e

    if shuffle is None:
        shuffle = train

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataset = MNIST(
        root=data_root,
        train=train,
        download=True,
        transform=_mnist_transforms(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_cifar10_dataloader(
    data_root: str = "./data/cifar10",
    *,
    train: bool = True,
    batch_size: int = 8,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
) -> DataLoader:
    """
    CIFAR-10 (3-channel RGB) DataLoader.

    Returns batches:
      - images: [B, 3, 32, 32] float32 (normalized)
      - labels: [B] int64
    """
    try:
        from torchvision.datasets import CIFAR10
    except Exception as e:
        raise ImportError(
            "torchvision is required for CIFAR-10. Install it via pip: pip install torchvision"
        ) from e

    if shuffle is None:
        shuffle = train

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataset = CIFAR10(
        root=data_root,
        train=train,
        download=True,
        transform=_cifar10_transforms(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_cifar10_dataloaders(
    data_root: str = "./data/cifar10",
    *,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Convenience: returns (train_loader, test_loader)."""
    train_loader = get_cifar10_dataloader(
        data_root,
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = get_cifar10_dataloader(
        data_root,
        train=False,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def get_mnist_dataloaders(
    data_root: str = "./data/MNIST",
    *,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Convenience: returns (train_loader, test_loader)."""
    train_loader = get_mnist_dataloader(
        data_root,
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = get_mnist_dataloader(
        data_root,
        train=False,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    # quick sanity check
    train_loader, test_loader = get_mnist_dataloaders(batch_size=8)
    x, y = next(iter(train_loader))
    print("train batch x:", tuple(x.shape), x.dtype, "y:", tuple(y.shape), y.dtype)
    x2, y2 = next(iter(test_loader))
    print("test batch x:", tuple(x2.shape), x2.dtype, "y:", tuple(y2.shape), y2.dtype)

    cifar_train, cifar_test = get_cifar10_dataloaders(batch_size=8)
    cx, cy = next(iter(cifar_train))
    print("cifar train batch x:", tuple(cx.shape), cx.dtype, "y:", tuple(cy.shape), cy.dtype)
    cx2, cy2 = next(iter(cifar_test))
    print("cifar test batch x:", tuple(cx2.shape), cx2.dtype, "y:", tuple(cy2.shape), cy2.dtype)

    anime_loader = get_anime_dataloader(batch_size=8, image_size=256, shuffle=True)
    ax, ap = next(iter(anime_loader))
    print("anime batch x:", tuple(ax.shape), ax.dtype, "paths[0]:", ap[0])
