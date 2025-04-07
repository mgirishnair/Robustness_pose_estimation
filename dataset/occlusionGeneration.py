import os
import json
import cv2
import random

annotations_path = "filtered_mpii_annotations.json"
image_root = "D:\\Delft\\Uni\\Mod3\\ComputerVision\\Project\\Data\\mpii_images\\images"
output_folder = "D:\\Delft\\Uni\\Mod3\\ComputerVision\\Project\\Data\\mpii_images_samples_occlusion_light"
os.makedirs(output_folder, exist_ok=True)

occlusion_level = "light"

if occlusion_level == "light":
    joints_to_occlude = 0
    patch_size_range = (0, 0)
elif occlusion_level == "moderate":
    joints_to_occlude = random.randint(2, 3)
    patch_size_range = (60, 100)
elif occlusion_level == "heavy":
    joints_to_occlude = random.randint(4, 6)
    patch_size_range = (80, 140)
else:
    raise ValueError("Invalid occlusion level")

def occlude_random_joints(image, keypoints, num_joints= joints_to_occlude, patch_size_range=patch_size_range):
    valid_kps = [(int(x), int(y)) for (x, y, v) in keypoints if v > 0]

    if not valid_kps:
        return image
    k = min(num_joints, len(valid_kps))
    selected_kps = random.sample(valid_kps, k)

    for x, y in selected_kps:
        patch_size = random.randint(*patch_size_range)
        half = patch_size // 2
        top_left = (max(0, x - half), max(0, y - half))
        bottom_right = (min(image.shape[1], x + half), min(image.shape[0], y + half))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)

    return image


with open(annotations_path, "r") as f:
    annotations = json.load(f)

for entry in annotations:
    img_name = entry["img_paths"]
    img_path = os.path.join(image_root, img_name)

    image = cv2.imread(img_path)
    if image is None:
        print(f"Could not load: {img_path}")
        continue

    keypoints = entry.get("joint_self", [])
    occluded_img = occlude_random_joints(image, keypoints)

    out_path = os.path.join(output_folder, os.path.basename(img_name))
    cv2.imwrite(out_path, occluded_img)

print("Done: Occluded random joints in all images.")
