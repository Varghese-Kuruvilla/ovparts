import argparse
import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate semantic segmentation metrics from multiple JSONL files.")
    parser.add_argument("--jsonls", nargs='+', required=True, help="Paths to the input JSONL files.")
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

def evaluate_single(jsonl_path, base_dir, ignore_label):
    data = []
    with open(jsonl_path, 'r') as f:
        try:
            content = json.load(f)
            if isinstance(content, list):
                data = content
            else:
                data = [content]
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    unique_labels = set()
    for entry in data:
        if "gt_parts" in entry:
            unique_labels.update(entry["gt_parts"].values())
        if "pred_parts" in entry:
            unique_labels.update(entry["pred_parts"].values())
    
    if "background" in unique_labels:
        unique_labels.remove("background")
    sorted_labels = sorted(list(unique_labels))
    label_to_id = {label: i for i, label in enumerate(sorted_labels)}
    num_classes = len(sorted_labels)
    
    if num_classes == 0:
        return None

    hist = np.zeros((num_classes, num_classes))
    
    for entry in tqdm(data, desc=f"Eval {os.path.basename(jsonl_path)}", leave=False):
        gt_path = os.path.join(base_dir, entry.get("gt_image_path", ""))
        pred_path = os.path.join(base_dir, entry.get("pred_image_path", ""))

        try:
            gt_img = np.array(Image.open(gt_path))
            pred_img = np.array(Image.open(pred_path))
        except Exception as e:
            print(f"Error loading images for entry {entry}: {e}")
            continue

        if gt_img.shape != pred_img.shape:
            if gt_img.shape[:2] != pred_img.shape[:2]:
                 print(f"Shape mismatch: {gt_path} {gt_img.shape} vs {pred_path} {pred_img.shape}. Skipping.")
                 continue

        gt_canonical = np.full(gt_img.shape[:2], ignore_label, dtype=np.int32)
        pred_canonical = np.full(pred_img.shape[:2], ignore_label, dtype=np.int32)

        gt_parts = entry.get("gt_parts", {})
        for pixel_val, label in gt_parts.items():
            if label in label_to_id:
                class_id = label_to_id[label]
                mask = (gt_img == int(pixel_val))
                if gt_img.ndim == 3:
                     pass
                gt_canonical[mask] = class_id

        pred_parts = entry.get("pred_parts", {})
        for pixel_val, label in pred_parts.items():
            if label in label_to_id:
                class_id = label_to_id[label]
                mask = (pred_img == int(pixel_val))
                pred_canonical[mask] = class_id
                
        hist += fast_hist(gt_canonical.flatten(), pred_canonical.flatten(), num_classes)

    acc = np.full(num_classes, np.nan, dtype=np.float64)
    iou = np.full(num_classes, np.nan, dtype=np.float64)
    tp = np.diag(hist).astype(np.float64)
    pos_gt = hist.sum(axis=1).astype(np.float64)
    
    pos_gt_sum = np.sum(pos_gt)
    if pos_gt_sum > 0:
        class_weights = pos_gt / pos_gt_sum
    else:
        class_weights = np.zeros_like(pos_gt)

    pos_pred = hist.sum(axis=0).astype(np.float64)
    acc_valid = pos_gt > 0
    acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    
    iou_valid = (pos_gt + pos_pred) > 0
    union = pos_gt + pos_pred - tp
    iou[acc_valid] = tp[acc_valid] / union[acc_valid]
    
    macc = np.sum(acc[acc_valid]) / np.sum(acc_valid) if np.sum(acc_valid) > 0 else 0.0
    miou = np.sum(iou[acc_valid]) / np.sum(iou_valid) if np.sum(iou_valid) > 0 else 0.0
    fiou = np.sum(iou[acc_valid] * class_weights[acc_valid]) if np.sum(acc_valid) > 0 else 0.0
    pacc = np.sum(tp) / pos_gt_sum if pos_gt_sum > 0 else 0.0

    res = {
        "mIoU": 100 * miou,
        "fwIoU": 100 * fiou,
        "mAcc": 100 * macc,
        "pAcc": 100 * pacc,
    }
    for i, name in enumerate(sorted_labels):
        res[f"IoU-{name}"] = 100 * iou[i] if acc_valid[i] else float('nan')
        res[f"Acc-{name}"] = 100 * acc[i] if acc_valid[i] else float('nan')
    return res

def main():
    args = parse_args()

    all_results = {}
    for path in args.jsonls:
        actual_path = path
        if not os.path.exists(actual_path):
            target_path = os.path.join(args.base_dir, path)
            if args.base_dir and os.path.exists(target_path):
                actual_path = target_path
            else:
                print(f"Warning: {path} not found.")
                continue

        print(f"Processing {path}...")
        res = evaluate_single(actual_path, args.base_dir, args.ignore_label)
        if res is not None:
            all_results[path] = res
        else:
            print(f"Warning: No valid classes found for {path}.")

    if not all_results:
        print("No valid results to display.")
        return

    # Print summary table
    print("\n" + "="*85)
    print(f"{'File':<40} | {'mIoU':<8} | {'fwIoU':<8} | {'mAcc':<8} | {'pAcc':<8}")
    print("-" * 85)
    for path, res in all_results.items():
        name = path
        if len(name) > 38:
            name = "..." + name[-35:]
        print(f"{name:<40} | {res['mIoU']:<8.2f} | {res['fwIoU']:<8.2f} | {res['mAcc']:<8.2f} | {res['pAcc']:<8.2f}")
    print("="*85 + "\n")

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
