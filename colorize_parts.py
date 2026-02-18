
import json
import os
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

def colorize_mask(mask_path, parts, output_path):
    try:
        mask = np.array(Image.open(mask_path))
    except Exception as e:
        print(f"Failed to open {mask_path}: {e}")
        return

    # Create an RGB image for visualization
    # Handle case where mask might be 2D or 3D (though likely 2D for segmentation)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0] # Assume single channel if 3D

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Define simple background/ignore
    BACKGROUND = 255
    
    # Sort unique values to ensure consistent coloring order for same set of parts
    unique_vals = sorted([v for v in np.unique(mask) if v != BACKGROUND])
    
    if not unique_vals:
        return

    # Generate distinct colors using HSV space for uniform distribution
    import colorsys
    
    n_colors = len(unique_vals)
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([int(c * 255) for c in rgb])
    
    # Shuffle colors to avoid adjacent parts having similar colors if indices are close
    random.shuffle(colors)
    
    for i, val in enumerate(unique_vals):
        color = colors[i]
        color_mask[mask == val] = color
    
    # Check output directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(color_mask).save(output_path)

def main():
    jsonl_path = '/archive/varghese/part_edit/data/ov_parts/PascalPart116/test.jsonl'
    
    # Create visualizations directory parallel to the annotations directory if possible,
    # or just a specific folder as requested. Since none requested, I'll create 'visualizations'
    # relative to the jsonl file location.
    base_dir = os.path.dirname(jsonl_path)
    output_dir = os.path.join(base_dir, 'visualizations_test')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading from {jsonl_path}")
    print(f"Saving visualizations to {output_dir}")
    
    entries = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    for entry in tqdm(entries):
        mask_path = entry['part_map_path']
        parts = entry['parts'] # Dict of "id": "name"
        
        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_path}")
            continue
            
        # Determine output filename
        # Use image filename but with png extension for mask visualization
        filename = os.path.basename(mask_path).replace('.png', '_colored.png')
        output_path = os.path.join(output_dir, filename)
        
        colorize_mask(mask_path, parts, output_path)
    
    print("Done.")

if __name__ == "__main__":
    main()
