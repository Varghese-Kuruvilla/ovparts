"""
Create JSONL splits for ADE20KPart234 with Object-Level Masks:
  1. train_base.jsonl       - Training split, base (seen) object annotations only
  2. val_base.jsonl         - Validation split, base (seen) annotations only
  3. val_novel.jsonl        - Validation split, novel (unseen) annotations only

Each entry includes:
  - image_path, part_map_path
  - obj_map_path: categorical mask (pixel value = category_id + 1)
  - class_name, parts, resolution
"""

import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils

DATA_ROOT = "/archive/varghese/part_edit/data/ov_parts/ADE20KPart234/ADE20KPart234"
OUTPUT_DIR = "/archive/varghese/part_edit/data/ov_parts/ADE20KPart234/ADE20KPart234"

OBJ_CLASS_NAMES = ['airplane', 'armchair', 'bed', 'bench', 'bookcase', 'bus', 'cabinet', 'car', 'chair', 'chandelier', 'chest of drawers', 'clock', 'coffee table', 'computer', 'cooking stove', 'desk', 'dishwasher', 'door', 
                   'fan', 'glass', 'kitchen island', 'lamp', 'light', 'microwave', 'minibike', 'ottoman', 'oven', 'person', 'pool table', 'refrigerator', 'sconce', 'shelf', 'sink', 'sofa', 'stool', 
                   'swivel chair', 'table', 'television receiver', 'toilet', 'traffic light', 'truck', 'van', 'wardrobe', 'washer']

OBJ_NOVEL_CLASS_NAMES = ['bench', 'bus', 'fan', 'desk', 'stool', 'truck', 'van', 'swivel chair', 'oven', 'ottoman', 'kitchen island']
OBJ_BASE_CLASS_NAMES = [c for c in OBJ_CLASS_NAMES if c not in OBJ_NOVEL_CLASS_NAMES]

PART_CLASS_NAMES = ["person's arm", "person's back", "person's foot", "person's gaze", "person's hand", "person's head", "person's leg", "person's neck", "person's torso", "door's door frame", "door's handle", "door's knob", 
               "door's panel", "clock's face", "clock's frame", "toilet's bowl", "toilet's cistern", "toilet's lid", "cabinet's door", "cabinet's drawer", "cabinet's front", "cabinet's shelf", 
               "cabinet's side", "cabinet's skirt", "cabinet's top", "sink's bowl", "sink's faucet", "sink's pedestal", "sink's tap", "sink's top", "lamp's arm", "lamp's base", "lamp's canopy", "lamp's column", 
               "lamp's cord", "lamp's highlight", "lamp's light source", "lamp's shade", "lamp's tube", "sconce's arm", "sconce's backplate", "sconce's highlight", "sconce's shade", "chair's apron",
               "chair's arm", "chair's back", "chair's base", "chair's leg", "chair's seat", "chair's seat cushion", "chair's skirt", "chair's stretcher", "chest of drawers's apron", "chest of drawers's door", "chest of drawers's drawer", 
               "chest of drawers's front", "chest of drawers's leg", "chandelier's arm", "chandelier's bulb", "chandelier's canopy", "chandelier's chain", "chandelier's cord", "chandelier's highlight", "chandelier's light source", "chandelier's shade",
               "bed's footboard", "bed's headboard", "bed's leg", "bed's side rail", "table's apron", "table's drawer", "table's leg", "table's shelf", "table's top", "table's wheel", "armchair's apron", "armchair's arm", "armchair's back", 
               "armchair's back pillow", "armchair's leg", "armchair's seat", "armchair's seat base", "armchair's seat cushion", "ottoman's back", "ottoman's leg", "ottoman's seat", "shelf's door", "shelf's drawer", "shelf's front", "shelf's shelf", 
               "swivel chair's back", "swivel chair's base", "swivel chair's seat", "swivel chair's wheel", "fan's blade", "fan's canopy", "fan's tube", "coffee table's leg", "coffee table's top", "stool's leg", "stool's seat", "sofa's arm", "sofa's back", 
               "sofa's back pillow", "sofa's leg", "sofa's seat base", "sofa's seat cushion", "sofa's skirt", "computer's computer case", "computer's keyboard", "computer's monitor", "computer's mouse", "desk's apron", "desk's door", "desk's drawer", "desk's leg",
               "desk's shelf", "desk's top", "wardrobe's door", "wardrobe's drawer", "wardrobe's front", "wardrobe's leg", "wardrobe's mirror", "wardrobe's top", "car's bumper", "car's door", "car's headlight", "car's hood", "car's license plate", "car's logo", 
               "car's mirror", "car's wheel", "car's window", "car's wiper", "bus's bumper", "bus's door", "bus's headlight", "bus's license plate", "bus's logo", "bus's mirror", "bus's wheel", "bus's window", "bus's wiper", "oven's window", "oven's door", 
               "oven's drawer", "oven's top", "cooking stove's burner", "cooking stove's button panel", "cooking stove's door", "cooking stove's drawer", "cooking stove's oven", "cooking stove's stove", "microwave's button panel", "microwave's door", "microwave's front",
               "microwave's side", "microwave's top", "microwave's window", "refrigerator's button panel", "refrigerator's door", "refrigerator's drawer", "refrigerator's side", "kitchen island's door", "kitchen island's drawer", "kitchen island's front", "kitchen island's side", 
               "kitchen island's top", "dishwasher's button panel", "dishwasher's handle", "dishwasher's skirt", "bookcase's door", "bookcase's drawer", "bookcase's front", "bookcase's side", "television receiver's base", "television receiver's buttons", "television receiver's frame",
               "television receiver's keys", "television receiver's screen", "television receiver's speaker", "glass's base", "glass's bowl", "glass's opening", "glass's stem", "pool table's bed", "pool table's leg", "pool table's pocket", "van's bumper", "van's door", "van's headlight", 
               "van's license plate", "van's logo", "van's mirror", "van's taillight", "van's wheel", "van's window", "van's wiper", "airplane's door", "airplane's fuselage", "airplane's landing gear", "airplane's propeller", "airplane's stabilizer", "airplane's turbine engine", 
               "airplane's wing", "truck's bumper", "truck's door", "truck's headlight", "truck's license plate", "truck's logo", "truck's mirror", "truck's wheel", "truck's window", "minibike's license plate", "minibike's mirror", "minibike's seat", "minibike's wheel", "washer's button panel", 
               "washer's door", "washer's front", "washer's side", "bench's arm", "bench's back", "bench's leg", "bench's seat", "traffic light's housing", "traffic light's pole", "light's aperture", "light's canopy", "light's diffusor", "light's highlight", "light's light source", "light's shade"]

def process_split(json_path, split_name, image_subdir, output_files):
    print(f"\nLoading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    img_id_to_info = {img['id']: img for img in data['images']}
    
    # Setup directory for object masks
    obj_mask_dir = os.path.join(DATA_ROOT, "annotations_detectron2_obj", image_subdir.split('/')[-1])
    os.makedirs(obj_mask_dir, exist_ok=True)
    
    annos_by_image = {}
    for anno in data['annotations']:
        img_id = anno['image_id']
        if img_id not in annos_by_image:
            annos_by_image[img_id] = []
        annos_by_image[img_id].append(anno)
    
    writers = {k: open(v, 'w') for k, v in output_files.items()}
    counts = {k: 0 for k in output_files}
    
    for img_id in tqdm(annos_by_image, desc=f"Processing {split_name}"):
        img_info = img_id_to_info.get(img_id)
        if not img_info: continue
        
        image_path = os.path.join(DATA_ROOT, image_subdir, img_info['file_name'])
        part_map_path = image_path.replace('images', 'annotations_detectron2_part').replace('.jpg', '.png')
        obj_map_base_path = image_path.replace('images', 'annotations_detectron2_obj').replace('.jpg', '.png')
        
        h, w = img_info['height'], img_info['width']
        resolution = [h, w]
        
        base_parts, novel_parts = {}, {}
        base_classes, novel_classes = set(), set()
        
        # Categorical object masks
        base_obj_mask = np.zeros((h, w), dtype=np.uint8)
        novel_obj_mask = np.zeros((h, w), dtype=np.uint8)
        
        for anno in annos_by_image[img_id]:
            obj_name = cat_id_to_name.get(anno['category_id'], "Unknown")
            cat_id = anno['category_id']
            part_ids = anno.get('part_category_id', [])
            
            # Decode object mask from RLE
            decoded_mask = mask_utils.decode(anno['segmentation'])
            
            if obj_name in OBJ_BASE_CLASS_NAMES:
                base_classes.add(obj_name)
                base_obj_mask[decoded_mask > 0] = cat_id + 1
                for pid in part_ids:
                    if pid < len(PART_CLASS_NAMES): base_parts[str(pid)] = PART_CLASS_NAMES[pid]
            elif obj_name in OBJ_NOVEL_CLASS_NAMES:
                novel_classes.add(obj_name)
                novel_obj_mask[decoded_mask > 0] = cat_id + 1
                for pid in part_ids:
                    if pid < len(PART_CLASS_NAMES): novel_parts[str(pid)] = PART_CLASS_NAMES[pid]
        
        def write_entry(target, classes, parts_dict, mask_data):
            if parts_dict and target in writers:
                # Save categorical object mask
                split_obj_map_path = obj_map_base_path if target == 'base' else obj_map_base_path.replace('.png', '_novel.png')
                Image.fromarray(mask_data).save(split_obj_map_path)
                
                record = {
                    "image_path": image_path,
                    "class_name": ", ".join(sorted(classes)),
                    "part_map_path": part_map_path,
                    "obj_map_path": split_obj_map_path,
                    "parts": parts_dict,
                    "resolution": resolution
                }
                writers[target].write(json.dumps(record) + '\n')
                counts[target] += 1

        write_entry('base', base_classes, base_parts, base_obj_mask)
        write_entry('novel', novel_classes, novel_parts, novel_obj_mask)
    
    for w in writers.values(): w.close()
    for k, v in counts.items(): print(f"  {k}: {v} entries -> {output_files[k]}")

def main():
    train_json = os.path.join(DATA_ROOT, "ade20k_instance_train.json")
    val_json = os.path.join(DATA_ROOT, "ade20k_instance_val.json")
    
    process_split(train_json, "Training", "images/training", {'base': os.path.join(OUTPUT_DIR, "ade20k_part234_train_base.jsonl")})
    process_split(val_json, "Validation", "images/validation", {
        'base': os.path.join(OUTPUT_DIR, "ade20k_part234_val_base.jsonl"),
        'novel': os.path.join(OUTPUT_DIR, "ade20k_part234_val_novel.jsonl")
    })

if __name__ == "__main__":
    main()
