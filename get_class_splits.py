import json
from collections import Counter

# The classes extracted directly from register_pascal_part_116.py
OBJ_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Load the JSON file
with open('/archive/varghese/part_edit/data/ov_parts/Datasets/PascalPart116/annotations_detectron2_part/train_obj_label_count.json', 'r') as f:
    data = json.load(f)

# Count occurrences of each class ID
class_counts = Counter()
for labels in data.values():
    class_counts.update(labels)

# Sort classes by frequency (descending)
sorted_counts = class_counts.most_common()

print(f"Total images analyzed: {len(data)}")
print("-" * 35)
print(f"{'Class Name':<15} | {'Image Count':<12}")
print("-" * 35)
for class_id, count in sorted_counts:
    class_name = OBJ_CLASS_NAMES[class_id] if class_id < len(OBJ_CLASS_NAMES) else f"Unknown ({class_id})"
    print(f"{class_name:<15} | {count:<12}")
print("-" * 35)
