# Reconstructed PointNet++: Towards fast and robust 3D perception

## Presentation
Please find the link to our presentation here:

## 1. Introduction
Due to the development of the self-driving industry, 3D perception has been an increasingly popular topic among researchers in recent years. Aiming at understanding complex real-world 3D environments, both model accuracy and inference speed are important factors for developing robust and reliable 3D perception systems.

One main branch of 3D feature learning is directly taking the point cloud as input and outputs for learning representation of the point cloud [1], [3]. This kind of point-based method preserves the spatial information of 3D data compared to convolutional methods but results in low processing speed due to the high computational complexity. The state-of-the-art method PVRCNN [3], has top performance in 3D detection among published methods on KITTI [4] benchmark, but cannot be processed on large point cloud inputs in real-time .

Considering such circumstances, we state our problem as improving the point-based methods to meet the real-time processing demands while maintaining the performance and reliability of the 3D perception models.

## 2. Related Work
PointNet [1]: PointNet is one of the pioneers’ works in the field of 3D data understanding. Instead of converting the point cloud to other representations, such as voxel and graph, PointNet directly takes point cloud as input and outputs either class labels for the entire point cloud or per point labels for segmentation tasks. The big advantage of processing point cloud directly is that the original information is well preserved. Also, compared with voxel representation, which carries voluminous and unnecessary data, point cloud representation is much more compact and thus requires less rendering resources. 

PointNet++ [2]: Although PointNet demonstrates excellent performance in classification, its capability for segmentation is limited because it does not capture local structures. PointNet++ solved this problem by introducing a new method called set abstraction. It can capture local features at multiple scales and achieves significantly better performance than the SOTAs.

PVRCNN [3]: Shi et al. [c] proposed a PointVoxel-RCNN (PV-RCNN) framework, which combines 3D convolutional network and PointNet-based set abstraction for the learning of point cloud features. The voxelized input points are fed into a 3D convolutional network to generate high-quality proposals. The learned voxel-wise features are then encoded into a small set of key points via a voxel set abstraction module to generate 3D bounding boxes.

All of these aforementioned methods rely on the Farthest Point Sampling(FPS) method for selecting key points from the original point cloud. This process is of high computation complexity and makes it impossible to process large point cloud inputs in real time. We proposed our Random Sampling(RS) based restructured model to speedup the forward process of such point-based methods and preserve the model performance to the maximum extent.

## 3. Dataset
For our project, we have identified two main data sources for model training:

ShapeNet: ShapeNet is a richly annotated, large-scale repository of shapes represented by 3D CAD models of objects. It contains 3D models from a multitude of semantic categories and organizes them under the WordNet taxonomy. It is a collection of datasets providing many semantic annotations for each 3D model.

ModelNet40: This dataset provides researchers in computer vision, computer graphics, robotics, and cognitive science, with a comprehensive clean collection of 3D CAD models for objects. It contains 12,311 pre-aligned shapes from 40 categories, which are split into 9,843 (80%) for training and 2,468 (20%) for testing. The CAD models are in Object File Format (OFF). To make this dataset work for our project, we sampled randomly from the CAD model surface to get ground truth point clouds with related labels.
