
import json
import os
from tqdm import tqdm

def main():
    input_jsonl = '/archive/varghese/part_edit/data/ov_parts/PascalPart116/test.jsonl'
    # Base directory for the newly created visualizations
    viz_dir = '/archive/varghese/part_edit/data/ov_parts/PascalPart116/visualizations_test'
    output_jsonl = '/archive/varghese/part_edit/data/ov_parts/PascalPart116/captioned_test.jsonl'
    
    print(f"Reading from {input_jsonl}")
    print(f"Writing to {output_jsonl}")
    
    entries = []
    skipped = 0
    
    with open(input_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
                
    output_entries = []
    
    for entry in tqdm(entries):
        image_path = entry['image_path']
        mask_path = entry['part_map_path']
        class_name = entry['class_name']
        parts_dict = entry['parts']
        
        # Determine the path of the colorized mask
        # It's in viz_dir and has _colored.png suffix
        mask_filename = os.path.basename(mask_path).replace('.png', '_colored.png')
        target_path = os.path.join(viz_dir, mask_filename)
        
        # Verify target exists
        if not os.path.exists(target_path):
            # Maybe the mask didn't have any valid parts and was skipped by colorize_parts.py?
            skipped += 1
            continue
            
        # Construct caption
        # "Generate a part-segmentation map of a <class_name> with <part1>, <part2>, ..."
        
        # Get list of unique part names
        part_names = sorted(list(set(parts_dict.values())))
        
        if not part_names:
            parts_str = "no parts"
        else:
            parts_str = ", ".join(part_names)
            
        caption = f"Generate a part-segmentation map of a {class_name} with {parts_str}"
        
        new_entry = {
            "source": image_path,
            "target": target_path,
            "caption": caption
        }
        
        output_entries.append(new_entry)
        
    print(f"Processed {len(entries)} entries.")
    print(f"Skipped {skipped} entries (missing target file).")
    print(f"Writing {len(output_entries)} entries to {output_jsonl}...")
    
    with open(output_jsonl, 'w') as f:
        for entry in output_entries:
            f.write(json.dumps(entry) + '\n')
            
    print("Done.")

if __name__ == "__main__":
    main()
