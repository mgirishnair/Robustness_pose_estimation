import json
import numpy as np


def compute_pckh(gt_keypoints, pred_keypoints, head_sizes, threshold=0.5):

    if len(gt_keypoints.shape) != 3 or gt_keypoints.shape[2] != 2:
        raise ValueError("gt_keypoints must have shape (N, K, 2)")
    if len(pred_keypoints.shape) != 3 or pred_keypoints.shape[2] != 2:
        raise ValueError("pred_keypoints must have shape (N, K, 2)")
    if gt_keypoints.shape[0] != pred_keypoints.shape[0]:
        raise ValueError("Mismatch: Ground truth and predictions have different numbers of samples")

    N, K, _ = gt_keypoints.shape  # Number of samples, keypoints, (x,y)

    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=2)  # Shape (N, K)

    normalized_distances = distances / head_sizes[:, np.newaxis]  # Shape (N, K)

    correct_keypoints = (normalized_distances < threshold).sum()

    total_keypoints = N * K

    pckh_score = correct_keypoints / total_keypoints

    return pckh_score


def load_ground_truth(gt_json_file, pred_filenames):
    with open(gt_json_file, 'r') as f:
        gt_data = json.load(f)

    gt_keypoints, head_sizes = {}, {}

    for img in gt_data:
        img_name = img.get("image_name", "")
        if img_name not in pred_filenames:
            continue

        img_keypoints, img_head_sizes = [], []

        for person in img.get("annotations", []):
            keypoints_list = person.get("keypoints", [])
            keypoints = np.full((16, 2), np.nan, dtype=np.float32)

            for kp in keypoints_list:
                kp_id = kp.get("id")
                if kp_id is not None and 0 <= kp_id < 16:
                    keypoints[kp_id] = [kp.get("x", 0), kp.get("y", 0)]

            scale = person.get("scale", 1.0)
            if not isinstance(scale, (int, float)) or scale is None:
                scale = 1.0
            head_size = float(scale) * 200

            img_keypoints.append(keypoints)
            img_head_sizes.append(head_size)

        gt_keypoints[img_name] = np.array(img_keypoints, dtype=np.float32)
        head_sizes[img_name] = np.array(img_head_sizes, dtype=np.float32)

    return gt_keypoints, head_sizes


def load_predictions(pred_json_file):
    with open(pred_json_file, 'r') as f:
        pred_data = json.load(f)

    pred_keypoints = {}

    for img_name in pred_data:
        image_preds = []
        predictions = pred_data[img_name][0]["predictions"][0]
        for person in predictions:
            keypoints = np.full((16, 2), np.nan, dtype=np.float32)
            for i, kp in enumerate(person.get("keypoints", [])):
                if i < 16 and isinstance(kp, list) and len(kp) == 2:
                    keypoints[i] = [kp[0], kp[1]]
            image_preds.append(keypoints)

        pred_keypoints[img_name] = np.array(image_preds, dtype=np.float32) if image_preds else np.empty((0, 16, 2),
                                                                                                        dtype=np.float32)

    return pred_keypoints


gt_json_filepath = 'D:\\Delft\\Uni\\Mod3\\ComputerVision\\Project\\Data\\mpii_anotations\\mpii_human_pose_v1_u12_2\\mpii_human_pose.json'
pred_json_filepath = 'D:\\Delft\\Uni\\Mod3\\ComputerVision\\Project\\Data\\mmpose_outputs\\pose_results.json'

pred_keypoints = load_predictions(pred_json_filepath)
pred_filenames = set(pred_keypoints.keys())

gt_keypoints, head_sizes = load_ground_truth(gt_json_filepath, pred_filenames)

aligned_gt, aligned_pred, aligned_head_sizes = [], [], []
for img_name in pred_keypoints:
    if img_name in gt_keypoints:
        aligned_gt.append(gt_keypoints[img_name])
        aligned_pred.append(pred_keypoints[img_name])
        aligned_head_sizes.append(head_sizes[img_name])

aligned_gt = np.concatenate(aligned_gt, axis=0) if aligned_gt else np.empty((0, 16, 2), dtype=np.float32)
aligned_pred = np.concatenate(aligned_pred, axis=0) if aligned_pred else np.empty((0, 16, 2), dtype=np.float32)
aligned_head_sizes = np.concatenate(aligned_head_sizes, axis=0) if aligned_head_sizes else np.empty((0,), dtype=np.float32)

pckh_score = compute_pckh(aligned_gt, aligned_pred, aligned_head_sizes) if aligned_gt.shape[0] > 0 else 0.0
print(f"PCKh Score: {pckh_score}")

