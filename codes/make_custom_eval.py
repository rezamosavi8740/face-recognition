import os
import random
import csv
import re
from itertools import combinations
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from datasets import Dataset, Image

def create_verification_csv(
    root_dir,
    output_csv,
    num_pairs_per_person=None,
    negative_ratio=1,
    max_iters=None,
    filter_strict_format=False,
    one_positive_per_person=False,
    num_workers=20
):
    valid_image_pattern = re.compile(r"^\d+_\d+\.jpg$", re.IGNORECASE)
    all_images = []
    person_to_images = {}

    def collect_images_for_person(person_folder):
        person_path = os.path.join(root_dir, person_folder)
        if not os.path.isdir(person_path):
            return None, []
        images = []
        for img in os.listdir(person_path):
            if not img.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            if filter_strict_format and not valid_image_pattern.match(img):
                continue
            images.append(os.path.join(person_folder, img))
        if len(images) < 2:
            return None, []
        return person_folder, images

    person_folders = os.listdir(root_dir)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(collect_images_for_person, pf) for pf in person_folders]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting images per person (threaded)"):
            person, images = future.result()
            if person is not None:
                person_to_images[person] = images
                all_images.extend(images)

    seen_negatives = set()
    negative_count = 0
    positive_count = 0
    max_positive = float("inf") if max_iters is None else max_iters

    with open(output_csv, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["image1", "image2", "is_same"])

        for person, images in tqdm(person_to_images.items(), desc="Writing positive pairs"):
            if positive_count >= max_positive:
                break
            pos_combinations = combinations(images, 2)
            count = 0
            for pair in pos_combinations:
                csvwriter.writerow([pair[0], pair[1], True])
                positive_count += 1
                count += 1
                if one_positive_per_person or (num_pairs_per_person and count >= num_pairs_per_person):
                    break
                if positive_count >= max_positive:
                    break

        target_negative = positive_count * negative_ratio
        all_images_set = set(all_images)

        while negative_count < target_negative:
            img1, img2 = random.sample(all_images, 2)
            if os.path.dirname(img1) != os.path.dirname(img2):
                key = tuple(sorted([img1, img2]))
                if key not in seen_negatives:
                    seen_negatives.add(key)
                    csvwriter.writerow([img1, img2, False])
                    negative_count += 1

    print(f"\n✅ CSV file created at: {output_csv}")
    print(f"Total pairs written: {positive_count + negative_count}")
    print(f"Positive pairs: {positive_count}")
    print(f"Negative pairs: {negative_count}")

def create_verification_dataset(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    image_paths = []
    is_same_list = []
    index_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV rows"):
        image_paths.append(os.path.join(image_dir, row["image1"]))
        image_paths.append(os.path.join(image_dir, row["image2"]))
        is_same_list.append(row["is_same"])
        is_same_list.append(row["is_same"])
        index_list.append(idx)
        index_list.append(idx)

    dataset = Dataset.from_dict({
        "image": image_paths,
        "is_same": is_same_list,
        "index": index_list,
    })

    dataset = dataset.cast_column("image", Image())
    return dataset

if __name__ == "__main__":
    root_dir = "/home/user1/data/bank_1M_aug"
    output_csv = "/home/user1/newdata/1M_bank_val/pairs.csv"
    num_pairs_per_person = 2
    negative_ratio = 2
    max_iters = None  # or set like 100000 for test

    create_verification_csv(
        root_dir=root_dir,
        output_csv=output_csv,
        num_pairs_per_person=num_pairs_per_person,
        negative_ratio=negative_ratio,
        max_iters=max_iters,
        filter_strict_format=False,
        one_positive_per_person=False,
        num_workers=20
    )

    dataset = create_verification_dataset(output_csv, root_dir)
    dataset.save_to_disk("/home/user1/newdata/1M_bank_val")
