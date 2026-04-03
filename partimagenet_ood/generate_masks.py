import json
import os
import numpy as np
from PIL import Image, ImageDraw
import tqdm

BASE_DIR = "/archive/varghese/part_edit/data/ov_parts/partimagenet_ood"

def generate_masks_for_split(split):
    json_path = os.path.join(BASE_DIR, f"{split}.json")
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: img for img in data['images']}
    
    # Pre-render directories
    base_part_dir = os.path.join(BASE_DIR, "annotations_detectron2_part", split)
    base_obj_dir = os.path.join(BASE_DIR, "annotations_detectron2_obj", split)
    
    # Sort annotations by image_id
    from collections import defaultdict
    image_to_anns = defaultdict(list)
    for ann in data['annotations']:
        image_to_anns[ann['image_id']].append(ann)
        
    print(f"Generating masks for {len(images)} images in {split}...")
    
    for img_id, img_info in tqdm.tqdm(images.items()):
        file_name = img_info['file_name']
        synset = file_name.split('_')[0]
        
        # Ensure directories exist
        part_dir = os.path.join(base_part_dir, synset)
        obj_dir = os.path.join(base_obj_dir, synset)
        os.makedirs(part_dir, exist_ok=True)
        os.makedirs(obj_dir, exist_ok=True)
        
        w, h = img_info['width'], img_info['height']
        
        # Part mask: pixel value = category_id + 1 (0 is background)
        # Note: Some pipelines use category_id directly if it starts from 1. 
        # PartImageNet starts from 0 (Quadruped Head). So we use category_id + 1.
        part_mask = Image.new('L', (w, h), 0)
        draw_part = ImageDraw.Draw(part_mask)
        
        # Object mask: pixel value = 1 (where any part is)
        obj_mask = Image.new('L', (w, h), 0)
        draw_obj = ImageDraw.Draw(obj_mask)
        
        anns = image_to_anns.get(img_id, [])
        # Important: sort by area desc so smaller parts aren't obscured if overlapping?
        # Actually for part mask usually we just paint. Order matters if there's overlap.
        for ann in sorted(anns, key=lambda x: x.get('area', 0), reverse=True):
            cat_id = ann['category_id']
            # Render segmentation (list of polygons)
            if 'segmentation' in ann:
                for poly in ann['segmentation']:
                    if len(poly) >= 4: # Must have at least 2 points
                        draw_part.polygon(poly, fill=cat_id + 1)
                        draw_obj.polygon(poly, fill=1)
        
        # Save as PNG
        stem = os.path.splitext(file_name)[0]
        part_mask.save(os.path.join(part_dir, f"{stem}.png"))
        obj_mask.save(os.path.join(obj_dir, f"{stem}.png"))

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        generate_masks_for_split(split)
