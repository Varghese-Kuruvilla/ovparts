import json
import os

BASE_DIR = "/archive/varghese/part_edit/data/ov_parts/partimagenet_ood"
MAPPING_PATH = "/home/varghese/OV_PARTS/partimagenet_ood/imagenet_class_index.json"
OUTPUT_DIR = "/home/varghese/OV_PARTS/partimagenet_ood"

# Load mapping
with open(MAPPING_PATH, 'r') as f:
    raw_mapping = json.load(f)
    synset_to_name = {v[0]: v[1].replace('_', ' ') for v in raw_mapping.values()}

# User specified categories
base_categories_names = [
    "tiger", "giant panda", "leopard", "gazelle", "green mamba", 
    "green lizard", "Komodo dragon", "tree frog", "yawl", "pirate", 
    "barracouta", "goldfish", "killer whale", "albatross", "goose", 
    "garbage truck", "minibus", "ambulance", "mountain bike", "moped", 
    "gorilla", "orangutan", "beer bottle", "water bottle", "warplane"
]

novel_categories_names = [
    "ice bear", "impala", "golden retriever", "Indian cobra", 
    "box turtle", "American alligator", "schooner", "tench", 
    "bald eagle", "jeep", "school bus", "motor scooter", 
    "chimpanzee", "wine bottle", "airliner"
]

# Map names to synsets
name_to_synset = {v: k for k, v in synset_to_name.items()}

def get_synset(name):
    # Try direct match
    if name in name_to_synset:
        return name_to_synset[name]
    # Try normalized match
    for k, v in synset_to_name.items():
        if v.lower() == name.lower():
            return k
    return None

base_synsets = [get_synset(n) for n in base_categories_names]
novel_synsets = [get_synset(n) for n in novel_categories_names]

# Filter out Nones
base_synsets = [s for s in base_synsets if s]
novel_synsets = [s for s in novel_synsets if s]

def create_jsonl(split, target_synsets, output_filename):
    json_path = os.path.join(BASE_DIR, f"{split}.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat for cat in data.get('categories', [])}
    
    # Group annotations
    annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann)
    
    jsonl_lines = []
    
    for img_id, img_info in images.items():
        file_name = img_info['file_name']
        synset = file_name.split('_')[0]
        
        if synset in target_synsets:
            # Get parts mapping for this image
            # Format: { "pixel_val": "part_name of a object_name" }
            # But wait, what are the pixel values? 
            # In some pipelines, pixel values are category_ids or just indices.
            # I'll use category_id as the key.
            
            anns = annotations.get(img_id, [])
            parts_dict = {}
            obj_name = synset_to_name.get(synset, synset)
            
            # Use original category names from the dataset (e.g., 'Quadruped Head')
            for i, ann in enumerate(anns):
                cat_info = categories.get(ann['category_id'], {"name": "part"})
                part_name = cat_info['name'].lower()
                # Use cat_id + 1 as key, matching PNG pixel values
                parts_dict[str(ann['category_id'] + 1)] = f"{part_name} of a {obj_name}"
            
            image_path = os.path.join(BASE_DIR, split, synset, file_name)
            
            # Standard naming for masks (consistent with ADE20K style)
            stem = os.path.splitext(file_name)[0]
            part_map_path = os.path.join(BASE_DIR, "annotations_detectron2_part", split, synset, f"{stem}.png")
            obj_map_path = os.path.join(BASE_DIR, "annotations_detectron2_obj", split, synset, f"{stem}.png")

            line = {
                "image_path": image_path,
                "class_name": obj_name,
                "part_map_path": part_map_path,
                "obj_map_path": obj_map_path,
                "parts": parts_dict,
                "resolution": [img_info['height'], img_info['width']]
            }
            jsonl_lines.append(json.dumps(line))
            
    with open(os.path.join(OUTPUT_DIR, output_filename), 'w') as f:
        f.write('\n'.join(jsonl_lines) + '\n')
    
    print(f"Created {output_filename} with {len(jsonl_lines)} entries.")

# Create the three files
create_jsonl("train", base_synsets, "partimagenet_train_base.jsonl")
create_jsonl("val", base_synsets, "partimagenet_val_base.jsonl")
create_jsonl("val", novel_synsets, "partimagenet_val_novel.jsonl")
