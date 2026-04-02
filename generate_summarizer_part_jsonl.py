import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

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

def get_class_and_parts(mask, ignore_val=255, keep_unseen=False):
    """
    Analyzes the mask to determine object classes and their present parts.
    Returns:
        class_names (list): List of chosen object classes present in the mask (either seen only, or unseen only).
        parts (dict): Mapping from pixel value (str) to part name for ONLY chosen classes.
    """
    UNSEEN_CLASSES = ["bird", "car", "dog", "sheep", "motorbike"]
    
    unique_vals = np.unique(mask)
    unique_vals = [v for v in unique_vals if v != ignore_val]
    
    if len(unique_vals) == 0:
        return [], {}
        
    # Find all unique classes present in the mask
    present_classes = set()
    for val in unique_vals:
        if val >= len(CLASS_NAMES):
            continue
            
        full_name = CLASS_NAMES[val]
        # CLASS_NAMES contains "person's head", "car's wheel", etc.
        # It does NOT contain simply "person" or "car".
        # We must always extract the base class name before the "'s "
        if "'s " in full_name:
            cls_name = full_name.split("'s ")[0]
        else:
            cls_name = full_name # Fallback, though everything in CLASS_NAMES has 's 
            
        present_classes.add(cls_name)
    
    # Filter out classes based on behavior
    if keep_unseen:
        # Include ONLY unseen classes
        chosen_classes = [c for c in present_classes if c in UNSEEN_CLASSES]
    else:
        # Include ONLY seen classes
        chosen_classes = [c for c in present_classes if c not in UNSEEN_CLASSES]
    
    if not chosen_classes:
        return [], {}
    
    # Collect parts only for seen classes
    parts = {}
    for val in unique_vals:
        if val >= len(CLASS_NAMES):
            continue
        
        full_name = CLASS_NAMES[val]
        
        if "'s " in full_name:
            curr_cls = full_name.split("'s ")[0]
            part_name = full_name.split("'s ")[1]
            
            # Only keep part if it belongs to a seen class
            if curr_cls in chosen_classes:
                parts[str(val)] = part_name
        else:
            # Fallback: only keep if it matches a seen class exactly
            if full_name in chosen_classes:
                parts[str(val)] = full_name
            
    return chosen_classes, parts

def generate_jsonl(base_dir, output_file, split='train', keep_unseen=False):
    img_root = os.path.join(base_dir, 'images_512')
    ann_root = os.path.join(base_dir, 'annotations_512')
    
    entries = []
    kept_images = 0
    skipped_images = 0
    
    split_ann_dir = os.path.join(ann_root, split)
    split_img_dir = os.path.join(img_root, split)
    
    if not os.path.exists(split_ann_dir):
        print(f"Directory {split_ann_dir} does not exist!")
        return
        
    files = sorted([f for f in os.listdir(split_ann_dir) if f.endswith('.png')])
    print(f"Processing {len(files)} files in {split}...")
    print(f"Keeping ONLY Unseen Classes: {keep_unseen}")
    
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
            chosen_classes, parts = get_class_and_parts(mask, keep_unseen=keep_unseen)
            
            # If there are no seen classes (either empty or completely unseen), skip it
            if not chosen_classes:
                skipped_images += 1
                continue
                
            kept_images += 1
            entry = {
                "image_path": img_path,
                "class_name": chosen_classes, # Now a list of all chosen classes in the image
                "part_map_path": ann_path,
                "parts": parts, # Contains parts ONLY for chosen classes
                "split": split
            }
            entries.append(entry)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    print(f"Split: {split} | Kept Images: {kept_images} | Skipped Images (Filtered/Empty): {skipped_images}")

    print(f"Writing {len(entries)} entries to {output_file}...")
    with open(output_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSONL for Pascal Parts 116 dataset.")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Dataset split to process (train or val).')
    parser.add_argument('--keep_unseen', action='store_true', help='Flag to include ONLY unseen classes (bird, car, dog, sheep, motorbike) and exclude seen classes.')
    
    args = parser.parse_args()
    
    base_path = '/archive/varghese/part_edit/data/ov_parts/PascalPart116'
    
    unseen_suffix = "_with_unseen" if args.keep_unseen else ""
    output_jsonl = os.path.join(base_path, f'pascal_parts_116_{args.split}{unseen_suffix}.jsonl')
    
    generate_jsonl(base_path, output_jsonl, split=args.split, keep_unseen=args.keep_unseen)
    print(f"Saved to {output_jsonl}")
