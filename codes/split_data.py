import os
import shutil
import random
from tqdm import tqdm

src_dir = "/home/user1/data/model/data/clean/440_aligned"
dst_dir = "/home/user1/data/model/data/clean/eval_data"

os.makedirs(dst_dir, exist_ok=True)

all_ids = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

all_ids.sort()
random.seed(42)
random.shuffle(all_ids)

num_eval = max(1, int(0.10 * len(all_ids)))
selected_eval_ids = all_ids[:num_eval]

for identity in tqdm(selected_eval_ids, desc="Moving eval identities"):
    src_path = os.path.join(src_dir, identity)
    dst_path = os.path.join(dst_dir, identity)
    shutil.move(src_path, dst_path)
