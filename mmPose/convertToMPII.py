
import json
import numpy as np

with open("keypoint_results_COCO.json", "r") as f:
    coco_data = json.load(f)


def avg(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, min(p1[2], p2[2])]


def extrapolate_head_top(nose, left_eye, right_eye):
    eye_center = [(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2]
    dx, dy = nose[0] - eye_center[0], nose[1] - eye_center[1]
    head_top = [nose[0] + 1.5 * dx, nose[1] + 1.5 * dy, nose[2]]
    return head_top


mpii_data = []

for img_name, person_data in coco_data.items():
    for person_kps in person_data["keypoints"]:
        coco = person_kps
        mpii = [None] * 16

        mpii[0] = coco[16]  # right ankle
        mpii[1] = coco[14]  # right knee
        mpii[2] = coco[12]  # right hip
        mpii[3] = coco[11]  # left hip
        mpii[4] = coco[13]  # left knee
        mpii[5] = coco[15]  # left ankle
        mpii[6] = avg(coco[11], coco[12])  # pelvis
        mpii[7] = avg(coco[5], coco[6])  # thorax
        mpii[8] = coco[0]  # upper neck ~ nose
        mpii[9] = extrapolate_head_top(coco[0], coco[1], coco[2])  # head top
        mpii[10] = coco[10]  # right wrist
        mpii[11] = coco[8]  # right elbow
        mpii[12] = coco[6]  # right shoulder
        mpii[13] = coco[5]  # left shoulder
        mpii[14] = coco[7]  # left elbow
        mpii[15] = coco[9]  # left wrist

        mpii_data.append({
            "image": img_name,
            "joints": mpii
        })

with open("keypoint_results_MPII.json", "w") as f:
    json.dump(mpii_data, f, indent=2)
