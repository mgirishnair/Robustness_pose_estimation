import os
import json

image_folder = "D:\\Delft\\Uni\\Mod3\\ComputerVision\\Project\\Data\\mpii_images_samples"  
input_json_path = "mpii_annotations.json"
output_json_path = "filtered_mpii_annotations.json"

image_files = set(os.listdir(image_folder))
print(len(image_files))
with open(input_json_path, 'r') as f:
    data = json.load(f)

filtered_data = [entry for entry in data if entry.get("img_paths") in image_files]

with open(output_json_path, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"Filtered {len(filtered_data)} entries saved to '{output_json_path}'")

unique_img_paths = {entry["img_paths"] for entry in filtered_data}
print(f"Unique image paths: {len(unique_img_paths)}")
