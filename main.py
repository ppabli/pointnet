import os
import argparse
import torch
from train import train, inference, visualize_sample

def main():

	parser = argparse.ArgumentParser(description='LiDAR Point Cloud Classification System using KITTI')

	parser.add_argument('--action', type=str, required=True, choices=['train', 'test', 'visualize'], help='Action to perform')
	parser.add_argument('--data_dir', type=str, default='./data/kitti', help='KITTI dataset directory')
	parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save results')
	parser.add_argument('--model', type=str, default='pointnet', choices=['pointnet', 'pointnetpp'], help='Model to use')
	parser.add_argument('--model_path', type=str, default=None, help='Path to pre-trained model')
	parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
	parser.add_argument('--num_points', type=int, default=4096, help='Number of points to use per sample')
	parser.add_argument('--feature_transform', action='store_true', help='Use feature transformation for PointNet')
	parser.add_argument('--sample_idx', type=int, default=0, help='Index of the sample to visualize')

	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)

	if args.action == 'train':

		print(f"Training {args.model} model with data from {args.data_dir}")

		train_args = argparse.Namespace(
			mode='train',
			data_path=args.data_dir,
			output_dir=args.output_dir,
			model=args.model,
			num_classes=3,
			num_points=args.num_points,
			use_cuda=torch.cuda.is_available(),
			batch_size=args.batch_size,
			epochs=args.epochs,
			lr=args.learning_rate,
			weight_decay=1e-4,
			feature_transform=args.feature_transform,
			feature_transform_regularizer=0.001,
			checkpoint_interval=10
		)

		train(train_args)

	elif args.action == 'test':

		if args.model_path is None:

			print("Error: --model_path is required for testing")
			return

		print(f"Evaluating {args.model} model from {args.model_path}")

		test_args = argparse.Namespace(
			mode='inference',
			data_path=args.data_dir,
			output_dir=args.output_dir,
			model=args.model,
			num_classes=3,
			num_points=args.num_points,
			use_cuda=torch.cuda.is_available(),
			batch_size=args.batch_size,
			model_path=args.model_path,
			feature_transform=args.feature_transform
		)

		inference(test_args)

	elif args.action == 'visualize':

		if args.model_path is None:

			print("Error: --model_path is required for visualization")
			return

		print(f"Visualizing results using {args.model_path}")

		vis_args = argparse.Namespace(
			mode='visualize',
			data_path=args.data_dir,
			output_dir=args.output_dir,
			model=args.model,
			num_classes=3,
			num_points=args.num_points,
			use_cuda=torch.cuda.is_available(),
			model_path=args.model_path,
			feature_transform=args.feature_transform
		)

		visualize_sample(vis_args)

	else:

		print("Invalid action. Use 'train', 'test' or 'visualize'")
		return

if __name__ == "__main__":

	main()