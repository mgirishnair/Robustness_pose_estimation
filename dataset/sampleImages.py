import os
import json
import shutil

json_path = "mpii_annotations.json"
image_source_folder = "D:\\Delft\\Uni\\Mod3\\ComputerVision\\Project\\Data\\mpii_occlusion_images"
image_target_folder = "D:\\Delft\\Uni\\Mod3\\ComputerVision\\Project\\Data\\mpii_images_samples"
num_images = 1000

with open(json_path, 'r') as f:
    data = json.load(f)

unique_filenames = []
seen = set()

for entry in data:
    img_name = entry.get("img_paths")
    if img_name and img_name not in seen:
        unique_filenames.append(img_name)
        seen.add(img_name)
    if len(unique_filenames) >= num_images:
        break

print(f"Collected {len(unique_filenames)} unique filenames.")

os.makedirs(image_target_folder, exist_ok=True)

copied = 0
for filename in unique_filenames:
    src = os.path.join(image_source_folder, filename)
    dst = os.path.join(image_target_folder, filename)
    if os.path.exists(src):
        shutil.copy(src, dst)
        copied += 1
    else:
        print(f"Warning: {filename} not found in {image_source_folder}")

print(f"Copied {copied} images to '{image_target_folder}'")
