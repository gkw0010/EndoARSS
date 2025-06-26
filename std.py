import os
import numpy as np
from PIL import Image

def load_mask(path):
    mask = Image.open(path)
    return np.array(mask)

def iou_score(pred_mask, true_mask):
    assert pred_mask.shape == true_mask.shape
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 1.0  # 都为空，算作完美
    else:
        return intersection / union

def compute_metrics(gt_root, pred_root):
    #folders = ['1', '2', '3']
    folders = [str(i) for i in range(1, 13)]

    class_correct = {f: 0 for f in folders}
    class_total = {f: 0 for f in folders}
    seg_ious = {f: [] for f in folders}

    pred_img_to_class = {}
    for pred_cls in folders:
        pred_mask_dir = os.path.join(pred_root, pred_cls, 'mask')
        if not os.path.exists(pred_mask_dir):
            continue
        for f in os.listdir(pred_mask_dir):
            pred_img_to_class[f] = pred_cls

    for gt_cls in folders:
        gt_mask_dir = os.path.join(gt_root, gt_cls, 'mask')
        for f in os.listdir(gt_mask_dir):
            gt_mask_path = os.path.join(gt_mask_dir, f)

            true_label = gt_cls
            class_total[true_label] += 1

            pred_label = pred_img_to_class.get(f, None)

            if pred_label == true_label:
                class_correct[true_label] += 1

            pred_mask_path = os.path.join(pred_root, pred_label if pred_label else 'unknown', 'mask', f)
            if pred_label is None or not os.path.exists(pred_mask_path):
                seg_ious[true_label].append(0.0)
            else:
                pred_mask = load_mask(pred_mask_path)
                true_mask = load_mask(gt_mask_path)
                iou = iou_score(pred_mask, true_mask)
                seg_ious[true_label].append(iou)

    class_accs = []
    seg_iou_means = []
    for c in folders:
        acc = class_correct[c] / class_total[c] if class_total[c] > 0 else 0
        class_accs.append(acc)

        mean_iou = np.mean(seg_ious[c]) if seg_ious[c] else 0
        seg_iou_means.append(mean_iou)

        print(f"Class {c}: classification accuracy = {acc:.4f}, mean IoU = {mean_iou:.4f}")

    print(f"\nClassification accuracy std: {np.std(class_accs):.6f}")
    print(f"Segmentation mean IoU std: {np.std(seg_iou_means):.6f}")

    return class_accs, seg_iou_means

if __name__ == "__main__":
    gt_root = "/mnt/data1_hdd/wgk/libmtllast/datasets/lastdata/EndoVis-18-VQA_extended/test"
    pred_root = "/mnt/data1_hdd/wgk/libmtllast/work_dirs/test_MOLA_STCH_cls_w=1_seg_w=1_dinonet"

    compute_metrics(gt_root, pred_root)



