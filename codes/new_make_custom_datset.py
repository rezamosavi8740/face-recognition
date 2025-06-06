
import os
import re
import shutil
import random
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageFilter
from tqdm import tqdm

# Regex to capture leading digits as identity
ID_PATTERN = re.compile(r"^(\d+)")

def random_augment(img: Image.Image) -> Image.Image:
    """Apply a random combination of flip, rotate, and blur to the image."""
    # Random horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Random rotation between -30 and +30 degrees
    angle = random.uniform(-30, 30)
    img = img.rotate(angle, resample=Image.BILINEAR)
    # Random Gaussian blur
    radius = random.uniform(0.5, 1.5)
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


def process_image(rel_path: str, source_root: Path, output_root: Path, num_augments: int):
    """Copy original and save random augmentations for one image."""
    # Extract identity from filename
    fname = os.path.basename(rel_path)
    m = ID_PATTERN.match(fname)
    if not m:
        return  # skip files without numeric prefix
    pid = m.group(1)

    # Create destination directory for this identity
    dest_dir = output_root / pid
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy original image
    src_file = source_root / rel_path
    dst_file = dest_dir / fname
    shutil.copy(src_file, dst_file)

    # Open image for augmentations
    img = Image.open(src_file).convert("RGB")

    # Generate and save augmentations
    base, ext = os.path.splitext(fname)
    for i in range(num_augments):
        aug_img = random_augment(img)
        aug_name = f"{base}_aug{i}{ext}"
        aug_path = dest_dir / aug_name
        aug_img.save(aug_path)


def main(args):
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    num_augments = args.num_augments
    num_workers = args.num_workers

    # Gather all relative image paths from subfolders
    rel_paths = []
    for sub in os.listdir(source_root):
        subdir = source_root / sub
        if not subdir.is_dir():
            continue
        for fname in os.listdir(subdir):
            if not fname.lower().endswith((".jpg",".jpeg",".png")):
                continue
            rel_paths.append(os.path.join(sub, fname))

    # Process with ThreadPool
    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for rel in rel_paths:
            futures.append(executor.submit(
                process_image, rel, source_root, output_root, num_augments
            ))

        # Progress bar as tasks complete
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            pass

    print(f"\n✅ All images processed. Augmented data at: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize and augment face images by identity.")
    parser.add_argument("--source_root", required=True,
                        help="Parent folder containing subfolders of raw images.")
    parser.add_argument("--output_root", required=True,
                        help="Destination root for identity folders and augmentations.")
    parser.add_argument("--num_augments", type=int, default=4,
                        help="Number of random augmentations per image.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of threads for parallel processing.")
    args = parser.parse_args()
    main(args)
