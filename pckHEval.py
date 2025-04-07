import json
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist

def compute_head_size(joints):
    if joints[9][2] == 0 or joints[8][2] == 0:
        return None
    return np.linalg.norm(np.array(joints[9][:2]) - np.array(joints[8][:2]))

def compute_pckh_single(gt_kps, pred_kps, alpha=0.5):
    head_size = compute_head_size(gt_kps)
    if head_size is None or head_size < 1e-6:
        return None

    correct_per_joint = []
    for i in range(16):
        if gt_kps[i][2] == 0:
            correct_per_joint.append(None)
            continue
        dist = np.linalg.norm(np.array(gt_kps[i][:2]) - np.array(pred_kps[i][:2]))
        correct = dist < alpha * head_size
        correct_per_joint.append(correct)
    return correct_per_joint

def match_by_objpos(gt_list, pred_list, max_dist=30.0):
    gt_matched = [False] * len(gt_list)
    pred_matched = [False] * len(pred_list)
    matches = []

    if not gt_list or not pred_list:
        return matches

    gt_pos = np.array([gt["objpos"] for gt in gt_list])
    pred_pos = np.array([pr["objpos"] for pr in pred_list])
    dists = cdist(gt_pos, pred_pos)

    for gt_idx, row in enumerate(dists):
        pred_idx = np.argmin(row)
        if row[pred_idx] <= max_dist and not pred_matched[pred_idx]:
            matches.append((gt_list[gt_idx], pred_list[pred_idx]))
            gt_matched[gt_idx] = True
            pred_matched[pred_idx] = True

    return matches

def evaluate_pckh(gt_file, pred_file, alpha=0.5):
    with open(gt_file) as f:
        gt_data = json.load(f)
    with open(pred_file) as f:
        pred_data = json.load(f)

    def group_by_img(data):
        grouped = defaultdict(list)
        for ann in data:
            grouped[ann["img_paths"]].append(ann)
        return grouped

    gt_grouped = group_by_img(gt_data)
    pred_grouped = group_by_img(pred_data)

    joint_hits = defaultdict(int)
    joint_total = defaultdict(int)

    for img_name in gt_grouped:
        gt_list = gt_grouped[img_name]
        pred_list = pred_grouped.get(img_name, [])
        matches = match_by_objpos(gt_list, pred_list)

        for gt, pred in matches:
            result = compute_pckh_single(gt["joint_self"], pred["joint_self"], alpha)
            if result is None:
                continue
            for j, correct in enumerate(result):
                if correct is not None:
                    joint_total[j] += 1
                    if correct:
                        joint_hits[j] += 1

    joint_names = [
        "r_ankle", "r_knee", "r_hip", "l_hip", "l_knee", "l_ankle",
        "pelvis", "thorax", "upper_neck", "head_top",
        "r_wrist", "r_elbow", "r_shoulder",
        "l_shoulder", "l_elbow", "l_wrist"
    ]

    print("=== PCKh Evaluation with Matching ===")
    pckh_all = []
    for j in range(16):
        total = joint_total[j]
        correct = joint_hits[j]
        score = correct / total if total > 0 else 0.0
        pckh_all.append(score)
        print(f"{joint_names[j]:<15}: {score:.3f} ({correct}/{total})")

    print(f"\nMean PCKh @ alpha={alpha}: {np.mean(pckh_all):.3f}")

if __name__ == "__main__":
    gt_file = "D:\\Delft\\Uni\\Mod3\\ComputerVision\\Project\\Data\\mpii_anotations\\filtered_mpii_annotations.json"
    # pred_file = "Heavy\\OpenPose\\converted_mpii_resultsOpenPose.json"  # openPose
    pred_file = "Heavy\\Yolo\\converted_keypoints_yolo.json"  # Yolo
    # pred_file = "Heavy\\mmPose\\converted_mpii_mmPose.json"  # mmPose
    evaluate_pckh(gt_file, pred_file, alpha=0.5)
