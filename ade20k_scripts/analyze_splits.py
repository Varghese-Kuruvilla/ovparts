import os
import json
from collections import Counter

DATA_ROOT = "/archive/varghese/part_edit/data/ov_parts/Datasets/ADE20KPart234"
TRAIN_JSON = os.path.join(DATA_ROOT, "ade20k_instance_train.json")
VAL_JSON = os.path.join(DATA_ROOT, "ade20k_instance_val.json")

OBJ_CLASS_NAMES = ['airplane', 'armchair', 'bed', 'bench', 'bookcase', 'bus', 'cabinet', 'car', 'chair', 'chandelier', 'chest of drawers', 'clock', 'coffee table', 'computer', 'cooking stove', 'desk', 'dishwasher', 'door', 
                   'fan', 'glass', 'kitchen island', 'lamp', 'light', 'microwave', 'minibike', 'ottoman', 'oven', 'person', 'pool table', 'refrigerator', 'sconce', 'shelf', 'sink', 'sofa', 'stool', 
                   'swivel chair', 'table', 'television receiver', 'toilet', 'traffic light', 'truck', 'van', 'wardrobe', 'washer']

OBJ_NOVEL_CLASS_NAMES = ['bench', 'bus', 'fan', 'desk', 'stool', 'truck', 'van', 'swivel chair', 'oven', 'ottoman', 'kitchen island']
OBJ_BASE_CLASS_NAMES = [c for c in OBJ_CLASS_NAMES if c not in OBJ_NOVEL_CLASS_NAMES]

PART_CLASS_NAMES = ["person's arm", "person's back", "person's foot", "person's gaze", "person's hand", "person's head", "person's leg", "person's neck", "person's torso", "door's door frame", "door's handle", "door's knob", 
               "door's panel", "clock's face", "clock's frame", "toilet's bowl", "toilet's cistern", "toilet's lid", "cabinet's door", "cabinet's drawer", "cabinet's front", "cabinet's shelf", 
               "cabinet's side", "cabinet's skirt", "cabinet's top", "sink's bowl", "sink's faucet", "sink's pedestal", "sink's tap", "sink's top", "lamp's arm", "lamp's base", "lamp's canopy", "lamp's column", 
               "lamp's cord", "lamp's highlight", "lamp's light source", "lamp's shade", "lamp's tube", "sconce's arm", "sconce's backplate", "sconce's highlight", "sconce's light source", "sconce's shade", "chair's apron",
               "chair's arm", "chair's back", "chair's base", "chair's leg", "chair's seat", "chair's seat cushion", "chair's skirt", "chair's stretcher", "chest of drawers's apron", "chest of drawers's door", "chest of drawers's drawer", 
               "chest of drawers's front", "chest of drawers's leg", "chandelier's arm", "chandelier's bulb", "chandelier's canopy", "chandelier's chain", "chandelier's cord", "chandelier's highlight", "chandelier's light source", "chandelier's shade",
               "bed's footboard", "bed's headboard", "bed's leg", "bed's side rail", "table's apron", "table's drawer", "table's leg", "table's shelf", "table's top", "table's wheel", "armchair's apron", "armchair's arm", "armchair's back", 
               "armchair's back pillow", "armchair's leg", "armchair's seat", "armchair's seat base", "armchair's seat cushion", "ottoman's back", "ottoman's leg", "ottoman's seat", "shelf's door", "shelf's drawer", "shelf's front", "shelf's shelf", 
               "swivel chair's back", "swivel chair's base", "swivel chair's seat", "swivel chair's wheel", "fan's blade", "fan's canopy", "fan's tube", "coffee table's leg", "coffee table's top", "stool's leg", "stool's seat", "sofa's arm", "sofa's back", 
               "sofa's back pillow", "sofa's leg", "sofa's seat base", "sofa's seat cushion", "sofa's skirt", "computer's computer case", "computer's keyboard", "computer's monitor", "computer's mouse", "desk's apron", "desk's door", "desk's drawer", "desk's leg",
               "desk's shelf", "desk's top", "wardrobe's door", "wardrobe's drawer", "wardrobe's front", "wardrobe's leg", "wardrobe's mirror", "wardrobe's top", "car's bumper", "car's door", "car's headlight", "car's hood", "car's license plate", "car's logo", 
               "car's mirror", "car's wheel", "car's window", "car's wiper", "bus's bumper", "bus's door", "bus's headlight", "bus's license plate", "bus's logo", "bus's mirror", "bus's wheel", "bus's window", "bus's wiper", "oven's button panel", "oven's door", 
               "oven's drawer", "oven's top", "cooking stove's burner", "cooking stove's button panel", "cooking stove's door", "cooking stove's drawer", "cooking stove's oven", "cooking stove's stove", "microwave's button panel", "microwave's door", "microwave's front",
               "microwave's side", "microwave's top", "microwave's window", "refrigerator's button panel", "refrigerator's door", "refrigerator's drawer", "refrigerator's side", "kitchen island's door", "kitchen island's drawer", "kitchen island's front", "kitchen island's side", 
               "kitchen island's top", "dishwasher's button panel", "dishwasher's handle", "dishwasher's skirt", "bookcase's door", "bookcase's drawer", "bookcase's front", "bookcase's side", "television receiver's base", "television receiver's buttons", "television receiver's frame",
               "television receiver's keys", "television receiver's screen", "television receiver's speaker", "glass's base", "glass's bowl", "glass's opening", "glass's stem", "pool table's bed", "pool table's leg", "pool table's pocket", "van's bumper", "van's door", "van's headlight", 
               "van's license plate", "van's logo", "van's mirror", "van's taillight", "van's wheel", "van's window", "van's wiper", "airplane's door", "airplane's fuselage", "airplane's landing gear", "airplane's propeller", "airplane's stabilizer", "airplane's turbine engine", 
               "airplane's wing", "truck's bumper", "truck's door", "truck's headlight", "truck's license plate", "truck's logo", "truck's mirror", "truck's wheel", "truck's window", "minibike's license plate", "minibike's mirror", "minibike's seat", "minibike's wheel", "washer's button panel", 
               "washer's door", "washer's front", "washer's side", "bench's arm", "bench's back", "bench's leg", "bench's seat", "traffic light's housing", "traffic light's pole", "light's aperture", "light's canopy", "light's diffusor", "light's highlight", "light's light source", "light's shade"]

PART_NOVEL_CLASSES = [name for name in PART_CLASS_NAMES if name.split("'s")[0] in OBJ_NOVEL_CLASS_NAMES]
PART_BASE_CLASSES = [name for name in PART_CLASS_NAMES if name.split("'s")[0] not in OBJ_NOVEL_CLASS_NAMES]

def analyze_json(json_path, split_name):
    print(f"\nAnalyzing {split_name} split: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    seen_obj_counts = Counter()
    unseen_obj_counts = Counter()
    
    seen_part_counts = Counter()
    unseen_part_counts = Counter()
    
    image_ids_seen = set()
    image_ids_unseen = set()
    
    for anno in data['annotations']:
        # Object level
        cat_name = cat_id_to_name.get(anno['category_id'], "Unknown")
        if cat_name in OBJ_NOVEL_CLASS_NAMES:
            unseen_obj_counts[cat_name] += 1
            image_ids_unseen.add(anno['image_id'])
        elif cat_name in OBJ_BASE_CLASS_NAMES:
            seen_obj_counts[cat_name] += 1
            image_ids_seen.add(anno['image_id'])
            
        # Part level
        if 'part_category_id' in anno:
            part_ids = anno['part_category_id']
            for pid in part_ids:
                if pid < len(PART_CLASS_NAMES):
                    pname = PART_CLASS_NAMES[pid]
                    if pname in PART_NOVEL_CLASSES:
                        unseen_part_counts[pname] += 1
                    else:
                        seen_part_counts[pname] += 1

    print("=" * 60)
    print(f"OBJECT SPLIT in {split_name}")
    print("-" * 60)
    print(f"Base Classes: {len(OBJ_BASE_CLASS_NAMES)} | Annotations: {sum(seen_obj_counts.values())}")
    print(f"Novel Classes: {len(OBJ_NOVEL_CLASS_NAMES)} | Annotations: {sum(unseen_obj_counts.values())}")
    
    print("\n" + "=" * 60)
    print(f"PART SPLIT in {split_name}")
    print("-" * 60)
    print(f"Base Part Classes: {len(PART_BASE_CLASSES)} | Annotations: {sum(seen_part_counts.values())}")
    print(f"Novel Part Classes: {len(PART_NOVEL_CLASSES)} | Annotations: {sum(unseen_part_counts.values())}")
    
    print("-" * 60)
    print(f"Total Base Images: {len(image_ids_seen)}")
    print(f"Total Novel Images: {len(image_ids_unseen)}")
    
    # Check for images that have both (if any)
    both = image_ids_seen.intersection(image_ids_unseen)
    if both:
        print(f"\nWarning: {len(both)} images have both base and novel objects.")

if __name__ == "__main__":
    analyze_json(TRAIN_JSON, "Training")
    analyze_json(VAL_JSON, "Validation")
