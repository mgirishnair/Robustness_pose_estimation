import os
import json
from pathlib import Path
from ultralytics import YOLO
from PIL import Image


image_folder = r"D:\Delft\Uni\Mod3\ComputerVision\Project\Data\mpii_images_samples_occlusion_light"

json_output_dir = Path(r"D:\Delft\Uni\Mod3\ComputerVision\Project\Data\Results\Light\Yolo\json")
json_output_dir.mkdir(parents=True, exist_ok=True)
output_json = json_output_dir / "keypoint_results.json"

visual_output_dir = Path(r"D:\Delft\Uni\Mod3\ComputerVision\Project\Data\Results\Light\Yolo\images")
visual_output_dir.mkdir(parents=True, exist_ok=True)

model = YOLO("checkpoints\\yolo11n-pose.pt")


image_paths = [os.path.join(image_folder, fname)
               for fname in os.listdir(image_folder)
               if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

#Inference
output_data = {}

for image_path in image_paths:
    image_name = os.path.basename(image_path)
    print(f"Processing: {image_name}")

    try:
        result = model(image_path)[0]
        keypoints_data = []

        if result.keypoints is not None and result.keypoints.xy is not None:
            for kp_xy, kp_conf in zip(result.keypoints.xy, result.keypoints.conf):
                person_keypoints = []
                for (x, y), conf in zip(kp_xy.tolist(), kp_conf.tolist()):
                    person_keypoints.append([x, y, conf])
                keypoints_data.append(person_keypoints)

        output_data[image_name] = {
            "keypoints": keypoints_data
        }

        plotted_np = result.plot()
        plotted_img = Image.fromarray(plotted_np)
        plotted_img.save(str(visual_output_dir / image_name))

    except Exception as e:
        print(f" Failed on {image_name}: {e}")


with open(output_json, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\nKeypoints saved to: {output_json}")
print(f" Visualizations saved to: {visual_output_dir}")
