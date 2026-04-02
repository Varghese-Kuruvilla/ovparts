import streamlit as st
import json
import os
import numpy as np
from PIL import Image

JSONL_ROOT = "/archive/varghese/part_edit/data/ov_parts/ADE20KPart234/ADE20KPart234"

# Use the same class names from the registration logic for consistent object legend
OBJ_CLASS_NAMES = ['airplane', 'armchair', 'bed', 'bench', 'bookcase', 'bus', 'cabinet', 'car', 'chair', 'chandelier', 'chest of drawers', 'clock', 'coffee table', 'computer', 'cooking stove', 'desk', 'dishwasher', 'door', 
                   'fan', 'glass', 'kitchen island', 'lamp', 'light', 'microwave', 'minibike', 'ottoman', 'oven', 'person', 'pool table', 'refrigerator', 'sconce', 'shelf', 'sink', 'sofa', 'stool', 
                   'swivel chair', 'table', 'television receiver', 'toilet', 'traffic light', 'truck', 'van', 'wardrobe', 'washer']

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

@st.cache_data
def get_colors(num_classes=256):
    np.random.seed(42)
    # Generate colorful values for everything
    colors = np.random.randint(60, 255, (num_classes + 1, 3), dtype=np.uint8)
    # Give Background a specific color ONLY if it's NOT a part, 
    # but here ID 0 is Arm, so we keep it colorful.
    # Instead, we can use a purple-ish theme for ID 0
    colors[0] = [200, 50, 50] # Explicit vibrant Red-ish for ID 0 (Arm)
    colors[num_classes] = [255, 255, 255] # White ignore
    return colors

def colorize_mask(mask, colors):
    """Apply the pre-generated colors to the mask."""
    # Ensure mask values are within bounds
    # Map high values (like 65535) to the last color index (usually White)
    max_idx = len(colors) - 1
    clamped_mask = np.where(mask < max_idx, mask, max_idx)
    return colors[clamped_mask]

def create_color_legend_html(label_dict, colors):
    """Creates an HTML legend with color boxes."""
    html = "<div style='line-height: 1.5; font-size: 0.9em;'>"
    max_idx = len(colors) - 1
    for val_str, name in label_dict.items():
        val = int(val_str)
        # Handle out of bounds mapping consistently with colorize_mask
        idx = val if val < max_idx else max_idx
        c = colors[idx]
        color_hex = '#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2])
        html += f"<div style='display: flex; align-items: center; margin-bottom: 4px;'>"
        html += f"<div style='width: 15px; height: 15px; background-color: {color_hex}; margin-right: 8px; border: 1px solid #555;'></div>"
        html += f"<span>ID {val}: {name}</span></div>"
    html += "</div>"
    return html

def main():
    st.set_page_config(layout="wide")
    st.title("ADE20K Part 234 JSONL Split Viewer")
    
    st.sidebar.header("Settings")
    jsonl_files = sorted([f for f in os.listdir(JSONL_ROOT) if f.endswith(".jsonl")])
    if not jsonl_files:
        st.error("No JSONL files found in data root.")
        return
        
    selected_jsonl = st.sidebar.selectbox("Select split", jsonl_files)
    items = load_jsonl(os.path.join(JSONL_ROOT, selected_jsonl))
    st.sidebar.info(f"Total entries: {len(items)}")
    
    idx = st.sidebar.number_input("Image Index", 0, len(items)-1, 0)
    item = items[idx]
    
    # Pre-generate colors
    colors = get_colors(num_classes=255)

    st.sidebar.divider()
    
    # Object Legend
    st.sidebar.write("### 🏷️ Object Legend")
    # Categorical masks use category_id + 1
    # We find which objects are actually mentioned in the 'class_name' string or list
    obj_legend_dict = {}
    classes = item['class_name']
    if isinstance(classes, str):
        classes = [c.strip() for c in classes.split(',')]
    
    for cname in classes:
        if cname in OBJ_CLASS_NAMES:
            cid = OBJ_CLASS_NAMES.index(cname)
            obj_legend_dict[str(cid + 1)] = cname
    
    st.sidebar.markdown(create_color_legend_html(obj_legend_dict, colors), unsafe_allow_html=True)
    
    st.sidebar.divider()

    # Parts Legend
    st.sidebar.write("### 🧩 Parts Legend")
    st.sidebar.markdown(create_color_legend_html(item['parts'], colors), unsafe_allow_html=True)

    # Main Display
    img = Image.open(item['image_path'])
    
    # Load Masks
    obj_mask_path = item.get('obj_map_path')
    obj_img_colored = None
    if obj_mask_path and os.path.exists(obj_mask_path):
        obj_mask = np.array(Image.open(obj_mask_path))
        obj_img_colored = colorize_mask(obj_mask, colors)
        
    part_mask = np.array(Image.open(item['part_map_path']))
    part_img_colored = colorize_mask(part_mask, colors)

    # Columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("#### Original Image")
        st.image(img, use_container_width=True)
    with col2:
        st.write("#### Object-Level Mask")
        if obj_img_colored is not None:
            st.image(obj_img_colored, use_container_width=True, caption="Categorical IDs")
        else:
            st.warning("Object mask missing")
    with col3:
        st.write("#### Part-Level Mask")
        st.image(part_img_colored, use_container_width=True, caption="Color-mapped by PID")
    
    st.success(f"File: {os.path.basename(item['image_path'])}")
    st.info(f"Full Path: {item['image_path']}")

if __name__ == "__main__":
    main()
