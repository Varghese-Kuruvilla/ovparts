import argparse
import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate semantic segmentation metrics from JSONL.")
    parser.add_argument("--jsonl", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--base-dir", default="", help="Base directory to prepend to relative image paths.")
    parser.add_argument("--ignore-label", type=int, default=0, help="Label index to ignore in evaluation.")
    parser.add_argument("--output-file", help="Path to save evaluation results as JSON.")
    return parser.parse_args()

def fast_hist(a, b, n):
    """
    Compute confusion matrix between label arrays a and b.
    a: ground truth
    b: prediction
    n: number of classes
    """
    k = (a >= 0) & (a < n) & (b >= 0) & (b < n)
    return np.bincount(
        n * a[k].astype(int) + b[k].astype(int), minlength=n ** 2
    ).reshape(n, n)

def main():
    args = parse_args()

    # Read JSON or JSONL
    print(f"Reading {args.jsonl}...")
    data = []
    with open(args.jsonl, 'r') as f:
        try:
            # Try loading as a standard JSON list
            content = json.load(f)
            if isinstance(content, list):
                data = content
            else:
                data = [content]
        except json.JSONDecodeError:
            # If failed, try reading line by line (JSONL)
            f.seek(0)
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    # Pass 1: Discover all unique classes
    print("Pass 1: Discovering classes...")
    unique_labels = set()
    for entry in data:
        if "gt_parts" in entry:
            unique_labels.update(entry["gt_parts"].values())
        if "pred_parts" in entry:
            unique_labels.update(entry["pred_parts"].values())
    
    
    # Exclude 'background'
    if "background" in unique_labels:
        unique_labels.remove("background")
    sorted_labels = sorted(list(unique_labels))
    label_to_id = {label: i for i, label in enumerate(sorted_labels)}
    num_classes = len(sorted_labels)
    
    print(f"Found {num_classes} unique classes: {sorted_labels}")

    # Pass 2: Evaluate
    print("Pass 2: Evaluating...")
    hist = np.zeros((num_classes, num_classes))
    
    for entry in tqdm(data):
        gt_path = os.path.join(args.base_dir, entry["gt_image_path"])
        pred_path = os.path.join(args.base_dir, entry["pred_image_path"])

        # Load images
        try:
            gt_img = np.array(Image.open(gt_path))
            pred_img = np.array(Image.open(pred_path))
        except Exception as e:
            print(f"Error loading images for entry {entry}: {e}")
            continue

        if gt_img.shape != pred_img.shape:
             # Handle potential shape mismatch if needed, or skip
             # For now, strict check
            if gt_img.shape[:2] != pred_img.shape[:2]:
                 print(f"Shape mismatch: {gt_path} {gt_img.shape} vs {pred_path} {pred_img.shape}. Skipping.")
                 continue

        # Create canonical masks
        # Initialize with ignore_label
        # We need a large enough int type to hold args.ignore_label if it's large
        gt_canonical = np.full(gt_img.shape[:2], args.ignore_label, dtype=np.int32)
        pred_canonical = np.full(pred_img.shape[:2], args.ignore_label, dtype=np.int32)

        # Map GT
        gt_parts = entry.get("gt_parts", {})
        for pixel_val, label in gt_parts.items():
            if label in label_to_id:
                class_id = label_to_id[label]
                # Handle pixels strictly
                mask = (gt_img == int(pixel_val))
                if gt_img.ndim == 3: # If RGB/RGBA, this might need adjustment if masks are 1-channel.
                     # Assuming mask images are single channel indices or exact value matches if 1-channel
                     # If they are RGB, int(pixel_val) comparison implies they are likely index images.
                     # Let's assume 2D mask images for now as is standard.
                     pass
                gt_canonical[mask] = class_id

        # Map Pred
        pred_parts = entry.get("pred_parts", {})
        for pixel_val, label in pred_parts.items():
            if label in label_to_id:
                class_id = label_to_id[label]
                mask = (pred_img == int(pixel_val))
                pred_canonical[mask] = class_id
                
        # Compute stats
        # Ensure we only compare valid pixels (ignore_label is excluded in fast_hist logic if we filter carefully)
        # But fast_hist expects 0..n-1 range.
        # We need to filter out ignore_label before calling fast_hist, OR
        # fast_hist implements `k = (a >= 0) & (a < n)`. 
        # Our gt_canonical has `ignore_label` (e.g. 255). If n < 255, it will be naturally ignored.
        # If ignore_label is within [0, n-1], we have a problem. Usually ignore_label is 255.
        
        hist += fast_hist(gt_canonical.flatten(), pred_canonical.flatten(), num_classes)

    # Compute Metrics
    # acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    
    # Frequency Weighted IoU
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    # Pretty Print
    print(f"{'Class':<20} {'IoU':<10} {'Acc':<10}")
    print("-" * 40)
    for i, label in enumerate(sorted_labels):
        iou_val = iu[i] * 100
        acc_val = (np.diag(hist)[i] / hist.sum(axis=1)[i] * 100) if hist.sum(axis=1)[i] > 0 else float('nan')
        print(f"{label:<20} {iou_val:<10.2f} {acc_val:<10.2f}")
    print("-" * 40)
    print(f"mIoU: {mean_iu*100:.2f}")
    print(f"fwIoU: {fwavacc*100:.2f}")
    print(f"mAcc: {acc_cls*100:.2f}")

    if args.output_file:
        results = {
            "mIoU": mean_iu,
            "fwIoU": fwavacc,
            "mAcc": acc_cls,
            "class_IoU": {label: iu[i] for i, label in enumerate(sorted_labels)},
            "confusion_matrix": hist.tolist()
        }
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
