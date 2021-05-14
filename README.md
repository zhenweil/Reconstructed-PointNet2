# Reconstructed PointNet++: Towards fast and robust 3D perception

## Presentation
Please find the link to our presentation here:

## 1. Introduction
Due to the development of the self-driving industry, 3D perception has been an increasingly popular topic among researchers in recent years. Aiming at understanding complex real-world 3D environments, both model accuracy and inference speed are important factors for developing robust and reliable 3D perception systems.

One main branch of 3D feature learning is directly taking the point cloud as input and outputs for learning representation of the point cloud [1], [3]. This kind of point-based method preserves the spatial information of 3D data compared to convolutional methods but results in low processing speed due to the high computational complexity. The state-of-the-art method PVRCNN [3], has top performance in 3D detection among published methods on KITTI [4] benchmark, but cannot be processed on large point cloud inputs in real-time .

Considering such circumstances, we state our problem as improving the point-based methods to meet the real-time processing demands while maintaining the performance and reliability of the 3D perception models.
