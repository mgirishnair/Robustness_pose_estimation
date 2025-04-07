import os
import json
import glob
import numpy as np


def compute_objpos(mpii_keypoints):
    visible_points = [kp[:2] for kp in mpii_keypoints if kp[2] > 0]
    if not visible_points:
        return [0.0, 0.0]
    visible_points = np.array(visible_points)
    center = visible_points.mean(axis=0)
    return [float(center[0]), float(center[1])]

def estimate_head_top(kps):
    reye = np.array([kps[15 * 3], kps[15 * 3 + 1], kps[15 * 3 + 2]])
    leye = np.array([kps[16 * 3], kps[16 * 3 + 1], kps[16 * 3 + 2]])
    nose = np.array([kps[0 * 3], kps[0 * 3 + 1], kps[0 * 3 + 2]])

    if reye[2] > 0 and leye[2] > 0:
        x = (reye[0] + leye[0]) / 2
        y = (reye[1] + leye[1]) / 2
        return [x, y, 1.0]
    elif nose[2] > 0:
        return [nose[0], nose[1], 1.0]
    else:
        return [0.0, 0.0, 0.0]

def convert_body25_to_mpii(keypoints):
    mpii_kps = [[0.0, 0.0, 0.0] for _ in range(16)]

    mapping = {
        11: 0,  # R ankle
        10: 1,  # R knee
        9: 2,   # R hip
        12: 3,  # L hip
        13: 4,  # L knee
        14: 5,  # L ankle
        8: 6,   # pelvis
        1: 7,   # thorax (neck)
        0: 8,   # upper neck ≈ nose
        4: 10,  # R wrist
        3: 11,  # R elbow
        2: 12,  # R shoulder
        5: 13,  # L shoulder
        6: 14,  # L elbow
        7: 15   # L wrist
    }

    for body25_idx, mpii_idx in mapping.items():
        x = keypoints[body25_idx * 3]
        y = keypoints[body25_idx * 3 + 1]
        conf = keypoints[body25_idx * 3 + 2]
        if conf > 0:
            mpii_kps[mpii_idx] = [x, y, 1.0]

    mpii_kps[9] = estimate_head_top(keypoints)

    return mpii_kps

def convert_all_body25_to_mpii(json_dir, output_path):
    all_annotations = []
    files = glob.glob(os.path.join(json_dir, "*_keypoints.json"))
    print(f"Found {len(files)} files...")

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        img_name = os.path.basename(file).replace("_keypoints.json", ".jpg")

        for person in data.get('people', []):
            kps = person.get('pose_keypoints_2d', [])
            if not kps:
                continue
            mpii_kps = convert_body25_to_mpii(kps)

            ann = {
                "dataset": "MPI",
                "isValidation": 0.0,
                "img_paths": img_name,
                "img_width": 0.0,
                "img_height": 0.0,
                "objpos":  compute_objpos(mpii_kps),
                "joint_self": mpii_kps,
                "scale_provided": 1.0,
                "joint_others": [],
                "scale_provided_other": 0.0,
                "objpos_other": [],
                "annolist_index": 0.0,
                "people_index": 0.0,
                "numOtherPeople": 0.0
            }

            all_annotations.append(ann)

    with open(output_path, 'w') as f:
        json.dump(all_annotations, f, indent=2)
    print(f"Saved {len(all_annotations)} converted annotations to {output_path}")

if __name__ == "__main__":
    input_json_folder = "json"
    output_json_path = "converted_mpii_resultsOpenPose.json"
    convert_all_body25_to_mpii(input_json_folder, output_json_path)
