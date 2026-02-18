
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# Copied from register_pascal_part_116.py
CLASS_NAMES = [
    "aeroplane's body", "aeroplane's stern", "aeroplane's wing", "aeroplane's tail", "aeroplane's engine", "aeroplane's wheel", 
    "bicycle's wheel", "bicycle's saddle", "bicycle's handlebar", "bicycle's chainwheel", "bicycle's headlight", 
    "bird's wing", "bird's tail", "bird's head", "bird's eye", "bird's beak", "bird's torso", "bird's neck", "bird's leg", "bird's foot", 
    "bottle's body", "bottle's cap", 
    "bus's wheel", "bus's headlight", "bus's front", "bus's side", "bus's back", "bus's roof", "bus's mirror", "bus's license plate", "bus's door", "bus's window", 
    "car's wheel", "car's headlight", "car's front", "car's side", "car's back", "car's roof", "car's mirror", "car's license plate", "car's door", "car's window", 
    "cat's tail", "cat's head", "cat's eye", "cat's torso", "cat's neck", "cat's leg", "cat's nose", "cat's paw", "cat's ear", 
    "cow's tail", "cow's head", "cow's eye", "cow's torso", "cow's neck", "cow's leg", "cow's ear", "cow's muzzle", "cow's horn", 
    "dog's tail", "dog's head", "dog's eye", "dog's torso", "dog's neck", "dog's leg", "dog's nose", "dog's paw", "dog's ear", "dog's muzzle", 
    "horse's tail", "horse's head", "horse's eye", "horse's torso", "horse's neck", "horse's leg", "horse's ear", "horse's muzzle", "horse's hoof", 
    "motorbike's wheel", "motorbike's saddle", "motorbike's handlebar", "motorbike's headlight", 
    "person's head", "person's eye", "person's torso", "person's neck", "person's leg", "person's foot", "person's nose", "person's ear", "person's eyebrow", "person's mouth", "person's hair", "person's lower arm", "person's upper arm", "person's hand",
    "pottedplant's pot", "pottedplant's plant", 
    "sheep's tail", "sheep's head", "sheep's eye", "sheep's torso", "sheep's neck", "sheep's leg", "sheep's ear", "sheep's muzzle", "sheep's horn", 
    "train's headlight", "train's head", "train's front", "train's side", "train's back", "train's roof", "train's coach", 
    "tvmonitor's screen"
]

def get_class_and_parts(mask, ignore_val=255):
    """
    Analyzes the mask to determine the object class and the present parts.
    Returns:
        class_name (str): The dominant object class.
        parts (dict): Mapping from pixel value (str) to part name.
    """
    unique_vals = np.unique(mask)
    unique_vals = [v for v in unique_vals if v != ignore_val]
    
    if len(unique_vals) == 0:
        return None, {}
        
    # Count pixels to find dominant class
    class_counts = {}
    
    for val in unique_vals:
        if val >= len(CLASS_NAMES):
            continue
            
        full_name = CLASS_NAMES[val]
        # Assume format "class's part"
        if "'s " in full_name:
            cls_name = full_name.split("'s ")[0]
        else:
            cls_name = full_name # Fallback
            
        # Count pixels for this value
        count = np.sum(mask == val)
        class_counts[cls_name] = class_counts.get(cls_name, 0) + count
        
    if not class_counts:
        return None, {}
        
    # Pick dominant class
    dominant_class = max(class_counts, key=class_counts.get)
    
    parts = {}
    for val in unique_vals:
        if val >= len(CLASS_NAMES):
            continue
        
        full_name = CLASS_NAMES[val]
        
        if "'s " in full_name:
            curr_cls = full_name.split("'s ")[0]
            part_name = full_name.split("'s ")[1]
            
            # Only keep part if it belongs to dominant class
            if curr_cls == dominant_class:
                parts[str(val)] = part_name
        else:
            # Fallback: only keep if it matches dominant class exactly
            if full_name == dominant_class:
                parts[str(val)] = full_name
            
    return dominant_class, parts

def generate_jsonl(base_dir, output_file):
    img_root = os.path.join(base_dir, 'images_512')
    ann_root = os.path.join(base_dir, 'annotations_512')
    
    entries = []
    
    # Traverse splits
    for split in ['train', 'val']:
        
        split_ann_dir = os.path.join(ann_root, split)
        split_img_dir = os.path.join(img_root, split)
        
        if not os.path.exists(split_ann_dir):
            continue
            
        files = sorted([f for f in os.listdir(split_ann_dir) if f.endswith('.png')])
        print(f"Processing {len(files)} files in {split}...")
        
        for f in tqdm(files):
            ann_path = os.path.join(split_ann_dir, f)
            img_name = f.replace('.png', '.jpg')
            img_path = os.path.join(split_img_dir, img_name)
            
            if not os.path.exists(img_path):
                # Try fallback just in case
                pass
                
            try:
                # Open with PIL
                mask = np.array(Image.open(ann_path))
                cls_name, parts = get_class_and_parts(mask)
                
                if cls_name:
                    entry = {
                        "image_path": img_path,
                        "class_name": cls_name,
                        "part_map_path": ann_path,
                        "parts": parts
                    }
                    entries.append(entry)
            except Exception as e:
                print(f"Error processing {f}: {e}")
    
    print(f"Writing {len(entries)} entries to {output_file}...")
    with open(output_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    base_path = '/archive/varghese/part_edit/data/ov_parts/PascalPart116'
    output_jsonl = os.path.join(base_path, 'pascal_parts_116.jsonl')
    
    generate_jsonl(base_path, output_jsonl)
    print(f"Saved to {output_jsonl}")
