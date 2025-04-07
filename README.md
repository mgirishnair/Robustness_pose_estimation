# Robustness_pose_estimation
Analysing the robustness of different pose estimation models 

## Dataset
The dataset used in this analysis is the MPII dataset which can be found in the following link: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/download. 

The dataset folder can be used to modify the dataset: 
- sampleImages.py - filters the whole dataset and samples 1000 images from it
- filtJSON.py - filters the annotations file (groundtruth) to only contain the keypoints for the smapled 1000 images.
- occlusionGeneration.py - allows to add occlusions to the images in different degree

## Models

### mmPose 
In order to run the inferencer, clone the following github: https://github.com/open-mmlab/mmpose. 
Then run the mmpose_inferencer.py after giving it the correct image data path. 

### openPose
In order to run the inferencer, git clone the following github: https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
Run the following command by changing the right data path: 
![image](https://github.com/user-attachments/assets/b1ed52fd-b946-458c-bbdb-06ff5d6a3d1c)

### yolo
In order to run the inferencer for yolo, download the checkpoints and run the yolo.py file with the right data paths. 

## Evaluation
The pckHeval.py can be used with the ground truth annotation file and the created mpii format output of all the models to obtain the PCKh evaluation of the models. 
