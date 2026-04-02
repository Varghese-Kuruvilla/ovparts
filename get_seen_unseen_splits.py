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

class_counts = Counter()
for labels in data.values():
    class_counts.update(labels)

seen_classes = []
unseen_classes = []
seen_count = 0
unseen_count = 0

for class_id, count in class_counts.items():
    if class_id < len(OBJ_CLASS_NAMES):
        class_name = OBJ_CLASS_NAMES[class_id]
        if class_name in OBJ_BASE_CLASS_NAMES:
            seen_classes.append((class_name, count))
            seen_count += count
        else:
            unseen_classes.append((class_name, count))
            unseen_count += count

print("=" * 40)
print(f"SEEN (BASE) OBJECT CLASSES ({len(seen_classes)} classes)")
print("=" * 40)
seen_classes.sort(key=lambda x: x[1], reverse=True)
for name, count in seen_classes:
    print(f"{name:<15} | {count:<12}")
print(f"-> Total Seen Image Count: {seen_count}")
print()

print("=" * 40)
print(f"UNSEEN (NOVEL) OBJECT CLASSES ({len(unseen_classes)} classes)")
print("=" * 40)
unseen_classes.sort(key=lambda x: x[1], reverse=True)
for name, count in unseen_classes:
    print(f"{name:<15} | {count:<12}")
print(f"-> Total Unseen Image Count: {unseen_count}")
print("=" * 40)
