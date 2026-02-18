
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

def get_part_bbox(mask, ignore_val=255):
    """Calculates the bounding box of parts in the mask."""
    # Parts are anything that is NOT the ignore value.
    
    rows = np.any(mask != ignore_val, axis=1)
    cols = np.any(mask != ignore_val, axis=0)
    
    if not np.any(rows):
        return None  # No parts found
        
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    return xmin, ymin, xmax, ymax

def crop_and_pad(img, mask, center, size=512, img_fill=(255, 255, 255), mask_fill=255):
    """Crops a fixed size region around the center and pads if necessary."""
    cx, cy = center
    half_size = size // 2
    
    h, w = img.shape[:2]
    
    # Calculate crop coordinates in the original image
    x1 = cx - half_size
    y1 = cy - half_size
    x2 = x1 + size
    y2 = y1 + size
    
    # Calculate overlap between crop and image
    img_x1 = max(0, x1)
    img_y1 = max(0, y1)
    img_x2 = min(w, x2)
    img_y2 = min(h, y2)
    
    # Calculate where to place the overlap in the output canvas
    out_x1 = img_x1 - x1
    out_y1 = img_y1 - y1
    out_x2 = out_x1 + (img_x2 - img_x1)
    out_y2 = out_y1 + (img_y2 - img_y1)
    
    # Initialize outputs
    out_img = np.full((size, size, 3), img_fill, dtype=img.dtype)
    out_mask = np.full((size, size), mask_fill, dtype=mask.dtype)
    
    if img_x2 > img_x1 and img_y2 > img_y1:
        out_img[out_y1:out_y2, out_x1:out_x2] = img[img_y1:img_y2, img_x1:img_x2]
        out_mask[out_y1:out_y2, out_x1:out_x2] = mask[img_y1:img_y2, img_x1:img_x2]
        
    return out_img, out_mask

def process_dataset(base_dir, split, output_base, limit=None):
    img_dir = os.path.join(base_dir, 'images', split)
    ann_dir = os.path.join(base_dir, 'annotations_detectron2_part', split)
    
    out_img_dir = os.path.join(base_dir, 'images_512', split)
    out_ann_dir = os.path.join(base_dir, 'annotations_512', split)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)
    
    # Filter for png files, ignoring dotfiles
    files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.png') and not f.startswith('.')])
    
    if limit:
        files = files[:limit]
        
    print(f"Processing {len(files)} files for split '{split}'...")
    
    for f in tqdm(files):
        ann_path = os.path.join(ann_dir, f)
        img_name = f.replace('.png', '.jpg') # Assuming jpg images based on checks
        img_path = os.path.join(img_dir, img_name)
        
        # Check if jpg exists, else try png if needed (though checks confirmed jpg)
        if not os.path.exists(img_path):
             # Try without extension swap if needed, but data check said annotations are png, images are jpg
             pass

        try:
            # Read Mask
            mask = np.array(Image.open(ann_path))
            
            # Read Image
            if os.path.exists(img_path):
                img = np.array(Image.open(img_path))
            else:
                print(f"Warning: Image not found {img_path}")
                continue
                
            # Get BBox of parts
            bbox = get_part_bbox(mask)
            
            if bbox is None:
                # If no parts found, center on image
                h, w = img.shape[:2]
                cx, cy = w // 2, h // 2
            else:
                xmin, ymin, xmax, ymax = bbox
                # Calculate Midpoint
                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2
                
            # Crop
            out_img_np, out_mask_np = crop_and_pad(img, mask, (cx, cy))
            
            # Save
            Image.fromarray(out_img_np).save(os.path.join(out_img_dir, img_name))
            Image.fromarray(out_mask_np).save(os.path.join(out_ann_dir, f))
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    # Hardcoded base path as requested by User
    base_path = '/archive/varghese/part_edit/data/ov_parts/PascalPart116'
    
    print(f"Reading from: {base_path}")
    print(f"Writing to: {base_path}/images_512 and {base_path}/annotations_512")

    # Process both splits
    # Check if 'val' exists, otherwise just process train or whatever is there
    if os.path.exists(os.path.join(base_path, 'images', 'train')):
        process_dataset(base_path, 'train', base_path)
    
    if os.path.exists(os.path.join(base_path, 'images', 'val')):
        process_dataset(base_path, 'val', base_path)
    
    print("Done.")
