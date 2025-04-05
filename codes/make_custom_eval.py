import os
import random
from itertools import combinations
import csv
import re
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, Image

def create_verification_csv(
    root_dir,
    output_csv,
    num_pairs_per_person=None,
    negative_ratio=1,
    max_iters=None,
    filter_strict_format=False,
    one_positive_per_person=False
):
    all_images = []
    person_to_images = {}
    valid_image_pattern = re.compile(r"^\d+_\d+\.jpg$", re.IGNORECASE)

    for person_folder in tqdm(os.listdir(root_dir), desc="Collecting images per person"):
        person_path = os.path.join(root_dir, person_folder)
        if os.path.isdir(person_path):
            images = []
            for img in os.listdir(person_path):
                if not img.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                if filter_strict_format and not valid_image_pattern.match(img):
                    continue
                images.append(os.path.join(person_folder, img))

            if len(images) < 2:
                continue
            all_images.extend(images)
            person_to_images[person_folder] = images

    positive_pairs = []
    for images in person_to_images.values():
        pos_combinations = list(combinations(images, 2))

        if one_positive_per_person:
            pos_combinations = random.sample(pos_combinations, 1)
        elif num_pairs_per_person is not None:
            pos_combinations = random.sample(pos_combinations, min(num_pairs_per_person, len(pos_combinations)))

        positive_pairs.extend(pos_combinations)

    num_negative_pairs = int(len(positive_pairs) * negative_ratio)
    negative_pairs = []
    while len(negative_pairs) < num_negative_pairs:
        img1, img2 = random.sample(all_images, 2)
        if os.path.dirname(img1) != os.path.dirname(img2):
            negative_pairs.append((img1, img2))

    total_iters = min(len(positive_pairs), len(negative_pairs))
    if max_iters is not None:
        total_iters = min(total_iters, max_iters)

    with open(output_csv, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["image1", "image2", "is_same"])
        for i in tqdm(range(total_iters), desc="Writing CSV pairs"):
            pos_pair = positive_pairs[i]
            neg_pair = negative_pairs[i]
            csvwriter.writerow([pos_pair[0], pos_pair[1], True])
            csvwriter.writerow([neg_pair[0], neg_pair[1], False])

    total_pairs = total_iters * 2
    print(f"\n✅ CSV file created at: {output_csv}")
    print(f"Total pairs written: {total_pairs}")
    print(f"Positive pairs: {total_iters}")
    print(f"Negative pairs: {total_iters}")

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
        index_list.append(idx * 2)
        index_list.append(idx * 2 + 1)

    dataset = Dataset.from_dict({
        "image": image_paths,
        "is_same": is_same_list,
        "index": index_list,
    })

    dataset = dataset.cast_column("image", Image())
    return dataset

root_dir = "/home/user1/data/model/data/clean/440_aligned"
output_csv = "/home/user1/newdata/440K_valid/pairs.csv"
num_pairs_per_person = None
negative_ratio = 1
max_iters = None

create_verification_csv(
    root_dir,
    output_csv,
    num_pairs_per_person=num_pairs_per_person,
    negative_ratio=negative_ratio,
    max_iters=max_iters,
    filter_strict_format=True,
    one_positive_per_person=True
)

dataset = create_verification_dataset(output_csv, root_dir)
dataset.save_to_disk("/home/user1/newdata/440K_valid")