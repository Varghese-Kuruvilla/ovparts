import json
from collections import Counter

# The classes extracted directly from register_pascal_part_116.py
OBJ_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# The base classes as defined in register_pascal_part_116.py (unseen excluded)
OBJ_BASE_CLASS_NAMES = [
    c for i, c in enumerate(OBJ_CLASS_NAMES) if c not in ["bird", "car", "dog", "sheep", "motorbike"]
]

with open('/archive/varghese/part_edit/data/ov_parts/Datasets/PascalPart116/annotations_detectron2_part/train_obj_label_count.json', 'r') as f:
    data = json.load(f)

only_base = 0
only_novel = 0
both = 0

for image, labels in data.items():
    has_base = False
    has_novel = False
    
    for class_id in labels:
        if class_id < len(OBJ_CLASS_NAMES):
            class_name = OBJ_CLASS_NAMES[class_id]
            if class_name in OBJ_BASE_CLASS_NAMES:
                has_base = True
            else:
                has_novel = True
                
    if has_base and has_novel:
        both += 1
    elif has_base:
        only_base += 1
    elif has_novel:
        only_novel += 1

print("--- IMAGE SPLITS ---")
print(f"Total Images: {len(data)}")
print(f"Images with ONLY Base (Seen) classes: {only_base}")
print(f"Images with ONLY Novel (Unseen) classes: {only_novel}")
print(f"Images with MIXED (Both Base and Novel) classes: {both}")
