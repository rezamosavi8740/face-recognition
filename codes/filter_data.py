import os
import shutil
import re
from tqdm import tqdm

source_root = "/home/user1/data/model/data/clean/440_aligned"
destination_root = "/home/user1/data/model/data/clean/440_aligned_base_image"

pattern = re.compile(r"^\d+-\d+\.jpg$", re.IGNORECASE)

for person_id in tqdm(os.listdir(source_root), desc="Processing identities"):
    person_path = os.path.join(source_root, person_id)
    if not os.path.isdir(person_path):
        continue

    dest_person_path = os.path.join(destination_root, person_id)
    os.makedirs(dest_person_path, exist_ok=True)

    for file_name in os.listdir(person_path):
        if pattern.match(file_name):
            src_file = os.path.join(person_path, file_name)
            dst_file = os.path.join(dest_person_path, file_name)
            shutil.copy2(src_file, dst_file)