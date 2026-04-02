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

PART_CLASS_NAMES = ["person's arm", "person's back", "person's foot", "person's gaze", "person's hand", "person's head", "person's leg", "person's neck", "person's torso", "door's door frame", "door's handle", "door's knob", 
               "door's panel", "clock's face", "clock's frame", "toilet's bowl", "toilet's cistern", "toilet's lid", "cabinet's door", "cabinet's drawer", "cabinet's front", "cabinet's shelf", 
               "cabinet's side", "cabinet's skirt", "cabinet's top", "sink's bowl", "sink's faucet", "sink's pedestal", "sink's tap", "sink's top", "lamp's arm", "lamp's base", "lamp's canopy", "lamp's column", 
               "lamp's cord", "lamp's highlight", "lamp's light source", "lamp's shade", "lamp's tube", "sconce's arm", "sconce's backplate", "sconce's highlight", "sconce's light source", "sconce's shade", "chair's apron",
               "chair's arm", "chair's back", "chair's base", "chair's leg", "chair's seat", "chair's seat cushion", "chair's skirt", "chair's stretcher", "chest of drawers's apron", "chest of drawers's door", "chest of drawers's drawer", 
               "chest of drawers's front", "chest of drawers's leg", "chandelier's arm", "chandelier's bulb", "chandelier's canopy", "chandelier's chain", "chandelier's cord", "chandelier's highlight", "chandelier's light source", "chandelier's shade",
               "bed's footboard", "bed's headboard", "bed's leg", "bed's side rail", "table's apron", "table's drawer", "table's leg", "table's shelf", "table's top", "table's wheel", "armchair's apron", "armchair's arm", "armchair's back", 
               "armchair's back pillow", "armchair's leg", "armchair's seat", "armchair's seat base", "armchair's seat cushion", "ottoman's back", "ottoman's leg", "ottoman's seat", "shelf's door", "shelf's drawer", "shelf's front", "shelf's shelf", 
               "swivel chair's back", "swivel chair's base", "swivel chair's seat", "swivel chair's wheel", "fan's blade", "fan's canopy", "fan's tube", "coffee table's leg", "coffee table's top", "stool's leg", "stool's seat", "sofa's arm", "sofa's back", 
               "sofa's back pillow", "sofa's leg", "sofa's seat base", "sofa's seat cushion", "sofa's skirt", "computer's computer case", "computer's keyboard", "computer's monitor", "computer's mouse", "desk's apron", "desk's door", "desk's drawer", "desk's leg",
               "desk's shelf", "desk's top", "wardrobe's door", "wardrobe's drawer", "wardrobe's front", "wardrobe's leg", "wardrobe's mirror", "wardrobe's top", "car's bumper", "car's door", "car's headlight", "car's hood", "car's license plate", "car's logo", 
               "car's mirror", "car's wheel", "car's window", "car's wiper", "bus's bumper", "bus's door", "bus's headlight", "bus's license plate", "bus's logo", "bus's mirror", "bus's wheel", "bus's window", "bus's wiper", "oven's button panel", "oven's door", 
               "oven's drawer", "oven's top", "cooking stove's burner", "cooking stove's button panel", "cooking stove's door", "cooking stove's drawer", "cooking stove's oven", "cooking stove's stove", "microwave's button panel", "microwave's door", "microwave's front",
               "microwave's side", "microwave's top", "microwave's window", "refrigerator's button panel", "refrigerator's door", "refrigerator's drawer", "refrigerator's side", "kitchen island's door", "kitchen island's drawer", "kitchen island's front", "kitchen island's side", 
               "kitchen island's top", "dishwasher's button panel", "dishwasher's handle", "dishwasher's skirt", "bookcase's door", "bookcase's drawer", "bookcase's front", "bookcase's side", "television receiver's base", "television receiver's buttons", "television receiver's frame",
               "television receiver's keys", "television receiver's screen", "television receiver's speaker", "glass's base", "glass's bowl", "glass's opening", "glass's stem", "pool table's bed", "pool table's leg", "pool table's pocket", "van's bumper", "van's door", "van's headlight", 
               "van's license plate", "van's logo", "van's mirror", "van's taillight", "van's wheel", "van's window", "van's wiper", "airplane's door", "airplane's fuselage", "airplane's landing gear", "airplane's propeller", "airplane's stabilizer", "airplane's turbine engine", 
               "airplane's wing", "truck's bumper", "truck's door", "truck's headlight", "truck's license plate", "truck's logo", "truck's mirror", "truck's wheel", "truck's window", "minibike's license plate", "minibike's mirror", "minibike's seat", "minibike's wheel", "washer's button panel", 
               "washer's door", "washer's front", "washer's side", "bench's arm", "bench's back", "bench's leg", "bench's seat", "traffic light's housing", "traffic light's pole", "light's aperture", "light's canopy", "light's diffusor", "light's highlight", "light's light source", "light's shade"]

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
