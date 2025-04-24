# Train PointNet
python main.py --action train --model pointnet --data_dir "/media/pablo/Disco programas/datasets/open3d/Kitti" --output_dir ./output --epochs 50 --batch_size 32 --num_points 1024 --feature_transform

# Train PointNet++
python main.py --action train --model pointnetpp --data_dir "/media/pablo/Disco programas/datasets/open3d/Kitti" --output_dir ./output --epochs 50 --batch_size 32 --num_points 1024

# Test PointNet
python main.py --action test --model pointnet --data_dir "/media/pablo/Disco programas/datasets/open3d/Kitti" --output_dir ./output --model_path ./output/pointnet_best.pth

# Test PointNet++
python main.py --action test --model pointnetpp --data_dir "/media/pablo/Disco programas/datasets/open3d/Kitti" --output_dir ./output --model_path ./output/pointnetpp_best.pth

# Visualize data sample PointNet
python main.py --action visualize --model pointnet --data_dir "/media/pablo/Disco programas/datasets/open3d/Kitti" --output_dir ./output --model_path ./output/pointnet_best.pth