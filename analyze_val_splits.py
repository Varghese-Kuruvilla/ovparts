import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from generate_summarizer_part_jsonl import CLASS_NAMES

UNSEEN_CLASSES = ["bird", "car", "dog", "sheep", "motorbike"]

split_ann_dir = '/archive/varghese/part_edit/data/ov_parts/PascalPart116/annotations_512/val'
files = sorted([f for f in os.listdir(split_ann_dir) if f.endswith('.png')])

only_base = 0
only_novel = 0
mixed = 0
empty = 0

for f in tqdm(files):
    ann_path = os.path.join(split_ann_dir, f)
    mask = np.array(Image.open(ann_path))
    unique_vals = np.unique(mask)
    unique_vals = [v for v in unique_vals if v != 255]
    
    present_classes = set()
    for val in unique_vals:
        if val >= len(CLASS_NAMES):
            continue
        full_name = CLASS_NAMES[val]
        if "'s " in full_name:
            cls_name = full_name.split("'s ")[0]
        else:
            cls_name = full_name
        present_classes.add(cls_name)
    
    has_seen = any(c for c in present_classes if c not in UNSEEN_CLASSES)
    has_unseen = any(c for c in present_classes if c in UNSEEN_CLASSES)
    
    if has_seen and has_unseen:
        mixed += 1
    elif has_seen:
        only_base += 1
    elif has_unseen:
        only_novel += 1
    else:
        empty += 1

print("--- VAL IMAGE SPLITS ---")
print(f"Total Images Analyzed: {len(files)}")
print(f"Images with ONLY Base (Seen) classes: {only_base}")
print(f"Images with ONLY Novel (Unseen) classes: {only_novel}")
print(f"Images with MIXED (Both Base and Novel) classes: {mixed}")
if empty > 0:
    print(f"Images skipped (Empty masks or unknown classes): {empty}")
