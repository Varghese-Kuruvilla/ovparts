import streamlit as st
import os
import numpy as np
from PIL import Image

st.set_page_config(layout="wide", page_title="PascalParts116 Viewer")

# Define paths
IMAGE_DIR = "/archive/varghese/part_edit/data/ov_parts/PascalPart116/images"
PART_DIR = "/archive/varghese/part_edit/data/ov_parts/PascalPart116/annotations_detectron2_part"
OBJ_DIR = "/archive/varghese/part_edit/data/ov_parts/PascalPart116/annotations_detectron2_obj"

@st.cache_data
def get_colors(num_classes=256):
    np.random.seed(42)
    colors = np.random.randint(50, 255, (num_classes + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0] # Background
    colors[num_classes] = [255, 255, 255] # Ignore/White
    return colors

def colorize_mask(mask, colors):
    if len(mask.shape) == 3:
        mask = mask[:,:,0] # if RGB, take first channel
    max_idx = len(colors) - 1
    clamped_mask = np.where(mask < max_idx, mask, max_idx)
    return colors[clamped_mask]

st.title("📂 PascalParts116 Viewer")
st.markdown("Visualizing images, object-level masks, and part-level masks from Detectron2 formats.")

# Sidebar Controls
st.sidebar.header("Controls")

splits = ["train", "val"]
selected_split = st.sidebar.selectbox("Select Split", splits)

# Get all valid image filenames
split_image_dir = os.path.join(IMAGE_DIR, selected_split)

if not os.path.exists(split_image_dir):
    st.error(f"Image directory for split '{selected_split}' not found at {split_image_dir}")
    st.stop()

# Ignore macOS hidden files
image_files = sorted([f for f in os.listdir(split_image_dir) if f.endswith(".jpg") and not f.startswith("._")])

if not image_files:
    st.warning(f"No JPG images found in {split_image_dir}")
    st.stop()

st.sidebar.info(f"Total Images: {len(image_files)}")

# Select Image
idx = st.sidebar.number_input("Select Image Index", min_value=0, max_value=len(image_files)-1, value=0)
selected_file = image_files[idx]
st.sidebar.text(f"Selected: {selected_file}")

# Compute paths
img_path = os.path.join(split_image_dir, selected_file)
# Masks are usually .png
base_name = os.path.splitext(selected_file)[0]
part_mask_path = os.path.join(PART_DIR, selected_split, base_name + ".png")
obj_mask_path = os.path.join(OBJ_DIR, selected_split, base_name + ".png")

# Load Image
img = Image.open(img_path).convert("RGB")

# Load and Colorize Masks
colors = get_colors(num_classes=255)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original Image")
    st.image(img, use_container_width=True)
    st.caption(f"Path: .../images/{selected_split}/{selected_file}")

with col2:
    st.subheader("Object level Annotations")
    if os.path.exists(obj_mask_path):
        obj_mask = np.array(Image.open(obj_mask_path))
        obj_colored = colorize_mask(obj_mask, colors)
        st.image(obj_colored, use_container_width=True)
        unique_obj_ids = np.unique(obj_mask)
        st.caption(f"Unique Object IDs in Mask: {unique_obj_ids.tolist()}")
    else:
        st.warning("Object mask missing or not found.")
        st.caption(f"Expected: {obj_mask_path}")

with col3:
    st.subheader("Part level Annotations")
    if os.path.exists(part_mask_path):
        part_mask = np.array(Image.open(part_mask_path))
        part_colored = colorize_mask(part_mask, colors)
        st.image(part_colored, use_container_width=True)
        unique_part_ids = np.unique(part_mask)
        st.caption(f"Unique Part IDs in Mask: {unique_part_ids.tolist()}")
    else:
        st.warning("Part mask missing or not found.")
        st.caption(f"Expected: {part_mask_path}")

# Add an option to overlay part mask
st.sidebar.divider()
show_overlay = st.sidebar.checkbox("Overlay Part Mask on Image", value=False)
if show_overlay and os.path.exists(part_mask_path):
    st.subheader("Part Mask Overlay")
    img_np = np.array(img)
    overlay = (img_np * 0.5 + part_colored * 0.5).astype(np.uint8)
    st.image(Image.fromarray(overlay), use_container_width=True)

