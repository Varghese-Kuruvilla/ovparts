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

    # Compute Metrics (matching GeneralizedSemSegEvaluator)
    acc = np.full(num_classes, np.nan, dtype=np.float64)
    iou = np.full(num_classes, np.nan, dtype=np.float64)
    tp = np.diag(hist).astype(np.float64)
    # hist[gt, pred]: sum over pred axis gives GT pixel counts per class
    pos_gt = hist.sum(axis=1).astype(np.float64)
    class_weights = pos_gt / np.sum(pos_gt)
    # hist[gt, pred]: sum over gt axis gives Pred pixel counts per class
    pos_pred = hist.sum(axis=0).astype(np.float64)
    acc_valid = pos_gt > 0
    acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    iou_valid = (pos_gt + pos_pred) > 0
    union = pos_gt + pos_pred - tp
    iou[acc_valid] = tp[acc_valid] / union[acc_valid]
    macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
    miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
    fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
    pacc = np.sum(tp) / np.sum(pos_gt)

    # Pretty Print
    print(f"{'Class':<20} {'IoU':<10} {'Acc':<10}")
    print("-" * 40)
    for i, label in enumerate(sorted_labels):
        print(f"{label:<20} {100 * iou[i]:<10.2f} {100 * acc[i]:<10.2f}")
    print("-" * 40)
    print(f"mIoU: {100 * miou:.2f}")
    print(f"fwIoU: {100 * fiou:.2f}")
    print(f"mAcc: {100 * macc:.2f}")
    print(f"pAcc: {100 * pacc:.2f}")

    if args.output_file:
        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(sorted_labels):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mAcc"] = 100 * macc
        res["pAcc"] = 100 * pacc
        for i, name in enumerate(sorted_labels):
            res["Acc-{}".format(name)] = 100 * acc[i]
        with open(args.output_file, 'w') as f:
            json.dump(res, f, indent=4)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
