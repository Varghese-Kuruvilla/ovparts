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

PART_CLASS_NAMES = ["arm of a person", "back of a person", "foot of a person", "gaze of a person", "hand of a person", "head of a person", "leg of a person", "neck of a person", "torso of a person", "door frame of a door", "handle of a door", "knob of a door", 
               "panel of a door", "face of a clock", "frame of a clock", "bowl of a toilet", "cistern of a toilet", "lid of a toilet", "door of a cabinet", "drawer of a cabinet", "front of a cabinet", "shelf of a cabinet", 
               "side of a cabinet", "skirt of a cabinet", "top of a cabinet", "bowl of a sink", "faucet of a sink", "pedestal of a sink", "tap of a sink", "top of a sink", "arm of a lamp", "base of a lamp", "canopy of a lamp", "column of a lamp", 
               "cord of a lamp", "highlight of a lamp", "light source of a lamp", "shade of a lamp", "tube of a lamp", "arm of a sconce", "backplate of a sconce", "highlight of a sconce", "light source of a sconce", "shade of a sconce", "apron of a chair",
               "arm of a chair", "back of a chair", "base of a chair", "leg of a chair", "seat of a chair", "seat cushion of a chair", "skirt of a chair", "stretcher of a chair", "apron of a chest of drawers", "door of a chest of drawers", "drawer of a chest of drawers", 
               "front of a chest of drawers", "leg of a chest of drawers", "arm of a chandelier", "bulb of a chandelier", "canopy of a chandelier", "chain of a chandelier", "cord of a chandelier", "highlight of a chandelier", "light source of a chandelier", "shade of a chandelier",
               "footboard of a bed", "headboard of a bed", "leg of a bed", "side rail of a bed", "apron of a table", "drawer of a table", "leg of a table", "shelf of a table", "top of a table", "wheel of a table", "apron of a armchair", "arm of a armchair", "back of a armchair", 
               "back pillow of a armchair", "leg of a armchair", "seat of a armchair", "seat base of a armchair", "seat cushion of a armchair", "back of a ottoman", "leg of a ottoman", "seat of a ottoman", "door of a shelf", "drawer of a shelf", "front of a shelf", "shelf of a shelf", 
               "back of a swivel chair", "base of a swivel chair", "seat of a swivel chair", "wheel of a swivel chair", "blade of a fan", "canopy of a fan", "tube of a fan", "leg of a coffee table", "top of a coffee table", "leg of a stool", "seat of a stool", "arm of a sofa", "back of a sofa", 
               "back pillow of a sofa", "leg of a sofa", "seat base of a sofa", "seat cushion of a sofa", "skirt of a sofa", "computer case of a computer", "keyboard of a computer", "monitor of a computer", "mouse of a computer", "apron of a desk", "door of a desk", "drawer of a desk", "leg of a desk",
               "shelf of a desk", "top of a desk", "door of a wardrobe", "drawer of a wardrobe", "front of a wardrobe", "leg of a wardrobe", "mirror of a wardrobe", "top of a wardrobe", "bumper of a car", "door of a car", "headlight of a car", "hood of a car", "license plate of a car", "logo of a car", 
               "mirror of a car", "wheel of a car", "window of a car", "wiper of a car", "bumper of a bus", "door of a bus", "headlight of a bus", "license plate of a bus", "logo of a bus", "mirror of a bus", "wheel of a bus", "window of a bus", "wiper of a bus", "button panel of a oven", "door of a oven", 
               "drawer of a oven", "top of a oven", "burner of a cooking stove", "button panel of a cooking stove", "door of a cooking stove", "drawer of a cooking stove", "oven of a cooking stove", "stove of a cooking stove", "button panel of a microwave", "door of a microwave", "front of a microwave",
               "side of a microwave", "top of a microwave", "window of a microwave", "button panel of a refrigerator", "door of a refrigerator", "drawer of a refrigerator", "side of a refrigerator", "door of a kitchen island", "drawer of a kitchen island", "front of a kitchen island", "side of a kitchen island", 
               "top of a kitchen island", "button panel of a dishwasher", "handle of a dishwasher", "skirt of a dishwasher", "door of a bookcase", "drawer of a bookcase", "front of a bookcase", "side of a bookcase", "base of a television receiver", "buttons of a television receiver", "frame of a television receiver",
               "keys of a television receiver", "screen of a television receiver", "speaker of a television receiver", "base of a glass", "bowl of a glass", "opening of a glass", "stem of a glass", "bed of a pool table", "leg of a pool table", "pocket of a pool table", "bumper of a van", "door of a van", "headlight of a van", 
               "license plate of a van", "logo of a van", "mirror of a van", "taillight of a van", "wheel of a van", "window of a van", "wiper of a van", "door of a airplane", "fuselage of a airplane", "landing gear of a airplane", "propeller of a airplane", "stabilizer of a airplane", "turbine engine of a airplane", 
               "wing of a airplane", "bumper of a truck", "door of a truck", "headlight of a truck", "license plate of a truck", "logo of a truck", "mirror of a truck", "wheel of a truck", "window of a truck", "license plate of a minibike", "mirror of a minibike", "seat of a minibike", "wheel of a minibike", "button panel of a washer", 
               "door of a washer", "front of a washer", "side of a washer", "arm of a bench", "back of a bench", "leg of a bench", "seat of a bench", "housing of a traffic light", "pole of a traffic light", "aperture of a light", "canopy of a light", "diffusor of a light", "highlight of a light", "light source of a light", "shade of a light"]

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
