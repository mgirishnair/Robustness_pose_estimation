import os
import json
from mmpose.apis import MMPoseInferencer
import cv2
from mmpose.visualization import local_visualizer

input_folder = "D:\\Delft\\Uni\\Mod3\\ComputerVision\\Project\\Data\\mpii_images_samples_occlusion_light"
output_folder = "D:\\Delft\\Uni\\Mod3\\ComputerVision\\Project\\Data\\Results\\Light\\mmPose\\images"
os.makedirs(output_folder, exist_ok=True)

inferencer = MMPoseInferencer('human')
# inferencer = MMPoseInferencer('rtmpose-tiny')

results_dict = {}

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        output_img_path = os.path.join(output_folder, filename)

        result_generator = inferencer(img_path, show=False, draw_out_dir=os.path.abspath(output_folder))

        results = list(result_generator)

        results_dict[filename] = results

        print(f"Processed: {filename} → Saved visualization & results.")

json_output_path = os.path.join("D:\\Delft\\Uni\\Mod3\\ComputerVision\\Project\\Data\\Results\\Light\\mmPose\\json", "pose_results.json")
with open(json_output_path, "w") as f:
    json.dump(results_dict, f, indent=4, default=float)

print(f"\n All images processed! Results saved to: {json_output_path}")
