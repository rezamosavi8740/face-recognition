from datasets import load_from_disk

ds = load_from_disk("/home/user1/newdata/eval_data")

# بررسی تعداد کل آیتم‌ها (هر جفت ۲ تصویره)
print(f"Total images: {len(ds)}")

# بررسی چند تا مثبت و منفی
from collections import Counter
print(Counter(ds["is_same"]))
