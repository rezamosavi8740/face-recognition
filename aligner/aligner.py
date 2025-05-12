#!/usr/bin/env python3
"""
Minimal demo: face detection ➜ face alignment
=============================================

• Reads a single image from disk
• Detects the biggest face using `do_face_detect`
• Aligns it with `do_align_dfa_onbbxo`
• Saves the aligned crop next to the original, prefixed with `aligned_`

Usage
-----
$ python detect_and_align.py path/to/photo.jpg --out_dir ./aligned
"""

from __future__ import annotations
import argparse
import os
import cv2  # OpenCV for image I/O

# ─── Your project helpers ──────────────────────────────────────────────────────
from preprocess.face_detect import do_face_detect            # face detector
from preprocess.dfa_allign import do_align_dfa_onbbxo as do_align  # aligner
# ───────────────────────────────────────────────────────────────────────────────


def load_image(path: str) -> "cv2.Mat":
    """Read an image from *path* and return it as a BGR NumPy array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def detect_and_align(img: "cv2.Mat") -> "cv2.Mat | None":
    """
    Run detection and alignment.

    Returns
    -------
    aligned_img : cv2.Mat | None
        • Aligned face as a NumPy array (BGR) if everything succeeds  
        • None if no face found or alignment failed
    """
    # 1️⃣ Face detection → bounding-box (your detector’s format)
    bbox = do_face_detect(img)
    if not bbox:
        print("No face detected.")
        return None

    # 2️⃣ Face alignment → aligned crop
    aligned = do_align(bbox, img, img.shape[:2])
    if aligned is None:
        print("Alignment failed.")
    return aligned


def save_image(img: "cv2.Mat", out_path: str) -> None:
    """Write *img* to *out_path* (creates directories if necessary)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, img)
    print(f"Aligned face saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect + align a single face.")
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument("--out_dir", default="aligned",
                        help="Directory to save the aligned face.")
    args = parser.parse_args()

    # Load → detect → align → save
    original = load_image(args.image)
    aligned  = detect_and_align(original)
    if aligned is not None:
        filename   = os.path.basename(args.image)
        aligned_fn = os.path.join(args.out_dir, f"aligned_{filename}")
        save_image(aligned, aligned_fn)


if __name__ == "__main__":   # Run when executed as a script
    main()
