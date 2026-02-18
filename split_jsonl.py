
import json
import os

def split_dataset(input_file, test_classes):
    base_dir = os.path.dirname(input_file)
    train_file = os.path.join(base_dir, 'train.jsonl')
    test_file = os.path.join(base_dir, 'test.jsonl')
    
    train_count = 0
    test_count = 0
    
    print(f"Reading from {input_file}...")
    print(f"Moving classes {test_classes} to {test_file}")
    
    try:
        with open(input_file, 'r') as f_in, \
             open(train_file, 'w') as f_train, \
             open(test_file, 'w') as f_test:
            
            for line in f_in:
                try:
                    entry = json.loads(line)
                    cls_name = entry.get('class_name')
                    
                    if cls_name in test_classes:
                        f_test.write(line)
                        test_count += 1
                    else:
                        f_train.write(line)
                        train_count += 1
                        
                except json.JSONDecodeError:
                    print(f"Skipping invalid line: {line[:50]}...")
                    
        print(f"Split complete.")
        print(f"Train entries: {train_count} -> {train_file}")
        print(f"Test entries:  {test_count} -> {test_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")

if __name__ == "__main__":
    input_path = '/archive/varghese/part_edit/data/ov_parts/PascalPart116/pascal_parts_116.jsonl'
    
    # Classes to move to test
    TEST_CLASSES = ["bird", "car", "dog", "sheep", "motorbike"]
    
    split_dataset(input_path, TEST_CLASSES)
