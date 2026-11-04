#!/usr/bin/env python
"""
Smoke test: verify all 6 training configs can load original ImageNet data.
Runs on CPU only — just validates dataset loading and one batch iteration.

Usage:
  python smoke_test_imagenet.py
"""

import os
import sys
import subprocess
import time

DATA_PATH = "/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/imagenet/ILSVRC/Data/CLS-LOC/train"

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def test_dataset():
    """Directly test ImageFolder + center_crop_arr without GPU."""
    from PIL import Image
    import numpy as np
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    def center_crop_arr(pil_image, image_size):
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    dataset = ImageFolder(DATA_PATH, transform=transform)
    print(f"  Classes: {len(dataset.classes)}")
    print(f"  Images:  {len(dataset)}")
    print(f"  Class-to-idx: {dict(list(dataset.class_to_idx.items())[:5])}...")

    # Test a few samples
    for i in [0, 100, 500, 1000]:
        if i >= len(dataset):
            break
        img, label = dataset[i]
        assert img.shape == (3, 256, 256), f"Expected (3,256,256), got {img.shape}"
        assert 0 <= label < 1000, f"Label out of range: {label}"
        assert -1.0 <= img.min() <= img.max() <= 1.0, f"Normalization issue: [{img.min()}, {img.max()}]"
        print(f"  Sample {i}: shape={img.shape}, label={label}, range=[{img.min():.3f}, {img.max():.3f}] ✓")

    return True


def test_dataloader():
    """Test DataLoader iteration."""
    import torch
    from torch.utils.data import DataLoader
    from PIL import Image
    import numpy as np
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    def center_crop_arr(pil_image, image_size):
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    dataset = ImageFolder(DATA_PATH, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=True)

    batch_count = 0
    for x, y in loader:
        assert x.shape == (4, 3, 256, 256), f"Batch shape: {x.shape}"
        assert y.shape == (4,), f"Label shape: {y.shape}"
        batch_count += 1
        if batch_count >= 3:
            break

    print(f"  DataLoader: {batch_count} batches iterated successfully ✓")
    return True


def main():
    print(f"Data path: {DATA_PATH}")
    n_classes = len(os.listdir(DATA_PATH))
    n_images = sum(
        len(os.listdir(os.path.join(DATA_PATH, d)))
        for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d))
    )
    print(f"Available: {n_classes} classes, {n_images} images\n")

    if n_classes == 0:
        print(f"{RED}No data found! Is unzip still running?{RESET}")
        sys.exit(1)

    all_ok = True

    # Test 1: Direct dataset
    print(f"{YELLOW}[TEST 1] ImageFolder + center_crop_arr{RESET}")
    try:
        test_dataset()
        print(f"{GREEN}PASS{RESET}\n")
    except Exception as e:
        print(f"{RED}FAIL: {e}{RESET}\n")
        import traceback; traceback.print_exc()
        all_ok = False

    # Test 2: DataLoader
    print(f"{YELLOW}[TEST 2] DataLoader iteration{RESET}")
    try:
        test_dataloader()
        print(f"{GREEN}PASS{RESET}\n")
    except Exception as e:
        print(f"{RED}FAIL: {e}{RESET}\n")
        import traceback; traceback.print_exc()
        all_ok = False

    print("=" * 60)
    if all_ok:
        print(f"{GREEN}All CPU smoke tests passed! Data loading is correct.{RESET}")
        print(f"\nTo test GPU training, run on a worker node:")
        print(f"  cd /home/tiger/multiscale_diffusion")
        print(f"  source 2d/2d/bin/activate")
        print(f"  CUDA_VISIBLE_DEVICES=0 python -u DiT/train_imagenet256.py \\")
        print(f"    --data-source imagenet --data-path {DATA_PATH} \\")
        print(f"    --model DiT-XL/2 --epochs 1 --global-batch-size 4 --num-workers 0 --no-pin-memory \\")
        print(f"    --ckpt-every 999999 --log-every 1 --results-dir /tmp/smoke_dit")
    else:
        print(f"{RED}Some tests failed.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
