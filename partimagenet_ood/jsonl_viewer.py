import streamlit as st
import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PartImageNet JSONL Viewer")
st.title("📂 PartImageNet JSONL Data Viewer")

BASE_DIR = "/home/varghese/OV_PARTS/partimagenet_ood"
JSONL_FILES = [
    "partimagenet_train_base.jsonl",
    "partimagenet_val_base.jsonl",
    "partimagenet_val_novel.jsonl"
]

@st.cache_data
def load_jsonl(filename):
    data = []
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Sidebar selection
selected_file = st.sidebar.selectbox("Select JSONL File", JSONL_FILES)
data = load_jsonl(selected_file)

if not data:
    st.error(f"Could not load data from {selected_file}")
    st.stop()

st.sidebar.success(f"Loaded {len(data)} entries")

# Filtering by class
all_classes = sorted(list(set(entry['class_name'] for entry in data)))
selected_class = st.sidebar.selectbox("Filter by Class", ["All"] + all_classes)

filtered_data = [d for d in data if d['class_name'] == selected_class] if selected_class != "All" else data
st.sidebar.write(f"Filtered: {len(filtered_data)} entries")

if not filtered_data:
    st.warning("No entries found for this class.")
    st.stop()

# Mask type selection
mask_type = st.sidebar.radio("View Mask Type", ["Part Mask", "Object Mask"])
mask_key = 'part_map_path' if mask_type == "Part Mask" else 'obj_map_path'

# Index selection
idx = st.sidebar.number_input("Select Entry Index", 0, len(filtered_data)-1, 0)
entry = filtered_data[idx]

# Display
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Original Image")
    img_path = entry['image_path']
    if os.path.exists(img_path):
        image = Image.open(img_path).convert("RGB")
        st.image(image, width="stretch")
    else:
        st.error(f"Image not found: {img_path}")

with col2:
    st.subheader(f"{mask_type}")
    mask_path = entry[mask_key]
    if os.path.exists(mask_path):
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        
        # Colorize mask
        if mask_type == "Part Mask":
            num_classes = 41 
            cmap = plt.get_cmap('tab20', num_classes)
        else:
            # Object mask is usually binary or few instances
            num_classes = 10
            cmap = plt.get_cmap('Set1', num_classes)
        
        colored_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
        unique_vals = np.unique(mask_array)
        
        for val in unique_vals:
            if val == 0: continue # background
            color = cmap(val % num_classes)[:3]
            colored_mask[mask_array == val] = [int(c * 255) for c in color]
        
        # Overlay if checkbox checked
        show_overlay = st.checkbox("Show Overlay on Original", value=True)
        if show_overlay and os.path.exists(img_path):
            img_np = np.array(image)
            overlay = (img_np * 0.5 + colored_mask * 0.5).astype(np.uint8)
            st.image(overlay, width="stretch")
        else:
            st.image(colored_mask, width="stretch")
    else:
        st.error(f"Mask not found: {mask_path}")

# Metadata
st.markdown("---")
m1, m2 = st.columns(2)
with m1:
    st.write(f"**Class Name:** :blue[{entry['class_name'].capitalize()}]")
    st.write(f"**Original Path:** `{img_path}`")
    st.write(f"**Mask Path:** `{mask_path}`")
    st.write(f"**Resolution:** {entry['resolution'][1]} x {entry['resolution'][0]} (W x H)")

with m2:
    st.write("**Parts Dictionary (Key = Pixel Value):**")
    parts = entry['parts']
    for p_id, p_name in parts.items():
        st.write(f"- `{p_id}`: **{p_name}**")

if st.checkbox("Show Raw JSON Record"):
    st.json(entry)
