import streamlit as st
import json
import os
from PIL import Image, ImageDraw
import numpy as np

# Set page config
st.set_page_config(page_title="PartImageNet OOD Viewer", layout="wide")

st.title("🖼️ PartImageNet OOD Dataset Viewer")

BASE_DIR = "/archive/varghese/part_edit/data/ov_parts/partimagenet_ood"
MAPPING_PATH = "/home/varghese/OV_PARTS/partimagenet_ood/imagenet_class_index.json"

@st.cache_data
def load_class_mapping():
    try:
        with open(MAPPING_PATH, 'r') as f:
            raw = json.load(f)
            # {index: [synset, name]} -> {synset: name}
            return {v[0]: v[1] for v in raw.values()}
    except Exception:
        return {}

class_mapping = load_class_mapping()
def load_data(split):
    json_path = os.path.join(BASE_DIR, f"{split}.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create mappings
    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat for cat in data.get('categories', [])}
    
    # Group annotations by image_id
    annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann)
        
    return images, annotations, categories

# Sidebar
st.sidebar.header("Settings")
split = st.sidebar.selectbox("Select Split", ["train", "test", "val"])

images_dict, annotations_dict, categories_dict = load_data(split)

# More sidebar controls
image_ids = sorted(list(images_dict.keys()))

# Search by filename
search_query = st.sidebar.text_input("Search by Filename (exact)")
if search_query:
    found_id = None
    for iid, info in images_dict.items():
        if info['file_name'] == search_query:
            found_id = iid
            break
    if found_id is not None:
        if found_id in image_ids:
            st.sidebar.success(f"Found at index {image_ids.index(found_id)}")
        else:
            st.sidebar.warning("Image exists but filtered out by category.")
    else:
        st.sidebar.error("Filename not found.")

# Filter by Object Class (Synset)
all_prefixes = sorted(list(set(images_dict[iid]['file_name'].split('_')[0] for iid in images_dict)))
obj_options = ["All"] + sorted([f"{class_mapping.get(p, p).replace('_', ' ').capitalize()} ({p})" for p in all_prefixes])
selected_obj = st.sidebar.selectbox("Filter by Object Class", obj_options)

if selected_obj != "All":
    prefix_id = selected_obj.split('(')[-1].strip(')')
    image_ids = [iid for iid in image_ids if images_dict[iid]['file_name'].startswith(prefix_id)]

# Filter by Part Category
cat_names = ["All"] + [cat['name'] for cat in categories_dict.values()]
selected_cat_name = st.sidebar.selectbox("Filter by Part Category", cat_names)

if selected_cat_name != "All":
    cat_id = [cid for cid, c in categories_dict.items() if c['name'] == selected_cat_name][0]
    # Filter image IDs that contain this category
    filtered_ids = []
    for img_id, anns in annotations_dict.items():
        if any(ann['category_id'] == cat_id for ann in anns):
            filtered_ids.append(img_id)
    image_ids = sorted(filtered_ids)
    st.sidebar.write(f"Filtered Images: {len(image_ids)}")

if not image_ids:
    st.warning("No images found for the selected category.")
    st.stop()

# Navigation
index = st.sidebar.number_input("Image Index", min_value=0, max_value=len(image_ids)-1, value=0)
img_id = image_ids[index]
img_info = images_dict[img_id]
anns = annotations_dict.get(img_id, [])

# Display image
file_name = img_info['file_name']
prefix = file_name.split('_')[0]
# Try with prefix first (ImageNet style)
img_path = os.path.join(BASE_DIR, split, prefix, file_name)
if not os.path.exists(img_path):
    # Try direct (just in case)
    img_path = os.path.join(BASE_DIR, split, file_name)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Image: {img_info['file_name']} (ID: {img_id})")
    try:
        img = Image.open(img_path).convert("RGB")
        
        # Draw annotations
        draw = ImageDraw.Draw(img, "RGBA")
        
        # Random colors for categories
        @st.cache_resource
        def get_colors():
            import random
            random.seed(42)
            colors = {}
            for cid in categories_dict.keys():
                colors[cid] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 100)
            return colors
        
        colors = get_colors()
        
        for ann in anns:
            cid = ann['category_id']
            color = colors.get(cid, (255, 0, 0, 100))
            
            # Draw segmentation
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for poly in ann['segmentation']:
                    if len(poly) >= 6: # At least 3 points
                        draw.polygon(poly, fill=color, outline=(color[0], color[1], color[2], 255))
            
            # Draw bbox (optional but helpful)
            # bbox = ann['bbox'] # [x, y, w, h]
            # draw.rectangle([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], outline=(255,255,255,150), width=2)

        st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")

with col2:
    st.subheader("Metadata")
    st.write(f"**Filename:** {img_info['file_name']}")
    
    # Get human-readable object name
    obj_name = class_mapping.get(prefix, f"Unknown ({prefix})")
    st.write(f"**Object Class:** :blue[{obj_name.replace('_', ' ').capitalize()}]")
    st.write(f"**Synset ID:** `{prefix}`")
    
    st.write(f"**Resolution:** {img_info['width']} x {img_info['height']}")
    st.write(f"**Part Annotations:** {len(anns)}")
    
    st.write("---")
    st.subheader("Part Classes in this Image")
    unique_cats = set(ann['category_id'] for ann in anns)
    for cid in unique_cats:
        c_info = categories_dict.get(cid, {"name": f"Unknown ({cid})", "supercategory": "None"})
        st.write(f"- **{c_info['name']}** ({c_info['supercategory']})")
    
    st.write("---")
    if st.checkbox("Show Raw JSON"):
        st.json({"image": img_info, "annotations": anns, "object_class": obj_name})

# Footer or additional info
st.markdown("---")
st.caption("Visualizer built for PartImageNet OOD Dataset analysis.")
