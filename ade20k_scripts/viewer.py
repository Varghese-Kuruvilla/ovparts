import streamlit as st
import json
import os
import numpy as np
from PIL import Image
import cv2

DATA_ROOT = "/archive/varghese/part_edit/data/ov_parts/Datasets/ADE20KPart234"
TRAIN_JSON = os.path.join(DATA_ROOT, "ade20k_instance_train.json")
VAL_JSON = os.path.join(DATA_ROOT, "ade20k_instance_val.json")

OBJ_CLASS_NAMES = ['airplane', 'armchair', 'bed', 'bench', 'bookcase', 'bus', 'cabinet', 'car', 'chair', 'chandelier', 'chest of drawers', 'clock', 'coffee table', 'computer', 'cooking stove', 'desk', 'dishwasher', 'door', 
                   'fan', 'glass', 'kitchen island', 'lamp', 'light', 'microwave', 'minibike', 'ottoman', 'oven', 'person', 'pool table', 'refrigerator', 'sconce', 'shelf', 'sink', 'sofa', 'stool', 
                   'swivel chair', 'table', 'television receiver', 'toilet', 'traffic light', 'truck', 'van', 'wardrobe', 'washer']

PART_CLASS_NAMES = ["arm of a person", "back of a person", "foot of a person", "gaze of a person", "hand of a person", "head of a person", "leg of a person", "neck of a person", "torso of a person", "door frame of a door", "handle of a door", "knob of a door", 
               "panel of a door", "face of a clock", "frame of a clock", "bowl of a toilet", "cistern of a toilet", "lid of a toilet", "door of a cabinet", "drawer of a cabinet", "front of a cabinet", "shelf of a cabinet", 
               "side of a cabinet", "skirt of a cabinet", "top of a cabinet", "bowl of a sink", "faucet of a sink", "pedestal of a sink", "tap of a sink", "top of a sink", "arm of a lamp", "base of a lamp", "canopy of a lamp", "column of a lamp", 
               "cord of a lamp", "highlight of a lamp", "light source of a lamp", "shade of a lamp", "tube of a lamp", "arm of a sconce", "backplate of a sconce", "highlight of a sconce", "light source of a sconce", "shade of a sconce", "apron of a chair",
               "arm of a chair", "back of a chair", "base of a chair", "leg of a chair", "seat of a chair", "seat cushion of a chair", "skirt of a chair", "stretcher of a chair", "apron of a chest of drawers", "door of a chest of drawers", "drawer of a chest of drawers", 
               "front of a chest of drawers", "leg of a chest of drawers", "arm of a chandelier", "bulb of a chandelier", "canopy of a chandelier", "chain of a chandelier", "cord of a chandelier", "highlight of a chandelier", "light source of a chandelier", "shade of a chandelier",
               "footboard of a bed", "headboard of a bed", "leg of a bed", "side rail of a bed", "apron of a table", "drawer of a table", "leg of a table", "shelf of a table", "top of a table", "wheel of a table", "apron of a armchair", "arm of a armchair", "back of a armchair", 
               "back pillow of a armchair", "leg of a armchair", "seat of a armchair", "seat base of a armchair", "seat cushion of a armchair", "back of a ottoman", "leg of a ottoman", "seat of a ottoman", "door of a shelf", "drawer of a shelf", "front of a shelf", "shelf of a shelf", 
               "back of a swivel chair", "base of a swivel chair", "seat of a swivel chair", "wheel of a swivel chair", "blade of a fan", "canopy of a fan", "tube of a fan", "leg of a coffee table", "top of a coffee table", "leg of a stool", "seat of a stool", "arm of a sofa", "back of a sofa", 
               "back pillow of a sofa", "leg of a sofa", "seat base of a sofa", "seat cushion of a sofa", "skirt of a sofa", "computer case of a computer", "keyboard of a computer", "monitor of a computer", "mouse of a computer", "apron of a desk", "door of a desk", "drawer of a desk", "leg of a desk",
               "shelf of a desk", "top of a desk", "door of a wardrobe", "drawer of a wardrobe", "front of a wardrobe", "leg of a wardrobe", "mirror of a wardrobe", "top of a wardrobe", "bumper of a car", "door of a car", "headlight of a car", "hood of a car", "license plate of a car", "logo of a car", 
               "mirror of a car", "wheel of a car", "window of a car", "wiper of a car", "bumper of a bus", "door of a bus", "headlight of a bus", "license plate of a bus", "logo of a bus", "mirror of a bus", "wheel of a bus", "window of a bus", "wiper of a bus", "button panel of a oven", "door of a oven", 
               "drawer of a oven", "top of a oven", "burner of a cooking stove", "button panel of a cooking stove", "door of a cooking stove", "drawer of a cooking stove", "oven of a cooking stove", "stove of a cooking stove", "button panel of a microwave", "door of a microwave", "front of a microwave",
               "side of a microwave", "top of a microwave", "window of a microwave", "button panel of a refrigerator", "door of a refrigerator", "drawer of a refrigerator", "side of a refrigerator", "door of a kitchen island", "drawer of a kitchen island", "front of a kitchen island", "side of a kitchen island", 
               "top of a kitchen island", "button panel of a dishwasher", "handle of a dishwasher", "skirt of a dishwasher", "door of a bookcase", "drawer of a bookcase", "front of a bookcase", "side of a bookcase", "base of a television receiver", "buttons of a television receiver", "frame of a television receiver",
               "keys of a television receiver", "screen of a television receiver", "speaker of a television receiver", "base of a glass", "bowl of a glass", "opening of a glass", "stem of a glass", "bed of a pool table", "leg of a pool table", "pocket of a pool table", "bumper of a van", "door of a van", "headlight of a van", 
               "license plate of a van", "logo of a van", "mirror of a van", "taillight of a van", "wheel of a van", "window of a van", "wiper of a van", "door of a airplane", "fuselage of a airplane", "landing gear of a airplane", "propeller of a airplane", "stabilizer of a airplane", "turbine engine of a airplane", 
               "wing of a airplane", "bumper of a truck", "door of a truck", "headlight of a truck", "license plate of a truck", "logo of a truck", "mirror of a truck", "wheel of a truck", "window of a truck", "license plate of a minibike", "mirror of a minibike", "seat of a minibike", "wheel of a minibike", "button panel of a washer", 
               "door of a washer", "front of a washer", "side of a washer", "arm of a bench", "back of a bench", "leg of a bench", "seat of a bench", "housing of a traffic light", "pole of a traffic light", "aperture of a light", "canopy of a light", "diffusor of a light", "highlight of a light", "light source of a light", "shade of a light"]

st.set_page_config(layout="wide")

@st.cache_data
def load_coco_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def get_image_info(data, image_id):
    for img in data['images']:
        if img['id'] == image_id:
            return img
    return None

def get_annotations(data, image_id):
    return [anno for anno in data['annotations'] if anno['image_id'] == image_id]

@st.cache_resource
def get_color_palette(n):
    # Fixed seed for consistent colors across runs
    np.random.seed(42)
    palette = np.random.randint(0, 255, (n, 3))
    palette[0] = [0, 0, 0] # Background is black
    return palette

def main():
    st.title("ADE20K Part 234 Viewer")
    
    split = st.sidebar.selectbox("Select Split", ["Training", "Validation"])
    json_path = TRAIN_JSON if split == "Training" else VAL_JSON
    
    data = load_coco_data(json_path)
    palette = get_color_palette(len(PART_CLASS_NAMES) + 1) # +1 for safely handling indices
    
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    image_ids = [img['id'] for img in data['images']]
    image_id = st.sidebar.selectbox("Select Image ID", image_ids)
    
    img_info = get_image_info(data, image_id)
    annos = get_annotations(data, image_id)
    
    if img_info:
        sub_folder = "images/training" if split == "Training" else "images/validation"
        img_path = os.path.join(DATA_ROOT, sub_folder, img_info['file_name'])
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            st.header(f"Image: {img_info['file_name']} (ID: {image_id})")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Original Image")
                
            seg_path = img_path.replace('images', 'annotations_detectron2_part').replace('jpg', 'png')
            if os.path.exists(seg_path):
                seg_img = Image.open(seg_path)
                seg_array = np.array(seg_img)
                
                # Use our consistent palette for colorization
                # Handle potential indices out of bounds just in case
                valid_mask = (seg_array < len(palette))
                colored_seg = np.zeros((*seg_array.shape, 3), dtype=np.uint8)
                colored_seg[valid_mask] = palette[seg_array[valid_mask]]
                
                with col2:
                    st.image(colored_seg, caption="Part Segmentation Map (Colorized)")
                    
                st.write(f"Segmentation Map Shape: {seg_array.shape}")
                unique_vals = np.unique(seg_array)
                
                # Legend with color coding
                st.subheader("Part Legend")
                for val in unique_vals:
                    if val == 0: continue # Skip background
                    
                    color = palette[val] if val < len(palette) else [127, 127, 127]
                    color_hex = '#%02x%02x%02x' % (color[0], color[1], color[2])
                    
                    name = PART_CLASS_NAMES[val] if val < len(PART_CLASS_NAMES) else f"Unknown ({val})"
                    
                    # Display colored square next to the name
                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: center; margin-bottom: 5px;">
                            <div style="width: 20px; height: 20px; background-color: {color_hex}; margin-right: 10px; border: 1px solid #ddd;"></div>
                            <span><strong>{val}</strong>: {name}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with st.expander(f"Show All {len(PART_CLASS_NAMES)} Part Classes"):
                    for i, name in enumerate(PART_CLASS_NAMES):
                        color = palette[i] if i < len(palette) else [127, 127, 127]
                        color_hex = '#%02x%02x%02x' % (color[0], color[1], color[2])
                        st.markdown(
                            f"""
                            <div style="display: flex; align-items: center; margin-bottom: 2px;">
                                <div style="width: 15px; height: 15px; background-color: {color_hex}; margin-right: 10px; border: 1px solid #eee;"></div>
                                <span style="font-size: 0.9em;">{i}: {name}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
            # Object annotations
            st.subheader("Object Annotations (COCO format)")
            for anno in annos:
                obj_name = categories.get(anno['category_id'], "Unknown")
                st.markdown(f"- **Object:** {obj_name} (Class ID: {anno['category_id']})")
                if 'part_category_id' in anno:
                    st.write(f"  - Parts IDs: {anno['part_category_id']}")
                if 'segmentation' in anno:
                    st.write(f"  - Has segmentation mask")
        else:
            st.error(f"Image not found at {img_path}")

if __name__ == "__main__":
    main()
