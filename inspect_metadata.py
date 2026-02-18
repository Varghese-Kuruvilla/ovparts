import os
import json

base_dir = '/archive/varghese/part_edit/data/ov_parts/PascalPart116'

try:
    files = os.listdir(base_dir)
    print(f"Files in {base_dir}:")
    for f in files:
        print(f" - {f}")

    # Check for json files
    json_files = [f for f in files if f.endswith('.json')]
    for jf in json_files:
        print(f"\nContent preview of {jf}:")
        try:
            with open(os.path.join(base_dir, jf), 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"List length: {len(data)}")
                    print(f"First item: {data[0]}")
                elif isinstance(data, dict):
                    print(f"Keys: {list(data.keys())}")
                    if 'categories' in data:
                        print(f"Categories sample: {data['categories'][:3]}")
                    if 'images' in data:
                        print(f"Images sample: {data['images'][:1]}")
                    if 'annotations' in data:
                        print(f"Annotations sample: {data['annotations'][:1]}")
        except Exception as e:
            print(f"Error reading {jf}: {e}")

except Exception as e:
    print(f"Error listing directory: {e}")
