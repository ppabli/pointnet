import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import open3d as o3d
import time
import csv
import gc
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, accuracy_score
import matplotlib.pyplot as plt
import tracemalloc

from pointnet import PointNet, feature_transform_regularizer
from pointnet_pp import PointNetPlusPlus
from kitti_dataset import get_kitti_object_dataloaders

def save_metrics_csv(metrics_dict, filepath):
	"""Save metrics to a CSV file

	Args:
		metrics_dict: Dictionary containing metrics data
		filepath: Path to save the CSV file
	"""
	os.makedirs(os.path.dirname(filepath), exist_ok=True)

	with open(filepath, 'w', newline='') as csvfile:

		fieldnames = list(metrics_dict.keys())
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		writer.writeheader()
		max_len = max([len(metrics_dict[k]) if isinstance(metrics_dict[k], list) else 1 for k in fieldnames])

		for key in fieldnames:

			if not isinstance(metrics_dict[key], list):

				metrics_dict[key] = [metrics_dict[key]] * max_len

		for i in range(max_len):

			row = {key: metrics_dict[key][i] if i < len(metrics_dict[key]) else None for key in fieldnames}
			writer.writerow(row)

	print(f"Metrics saved to {filepath}")

def train(args):
	"""Train the PointNet/PointNet++ model on the KITTI dataset"""

	device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
	print(f"Using device: {device}")

	train_loader, val_loader, _ = get_kitti_object_dataloaders(
		root_dir=args.data_path,
		batch_size=args.batch_size,
		num_points=args.num_points
	)

	if args.model == 'pointnet':

		model = PointNet(num_classes=args.num_classes, feature_transform=args.feature_transform)

	elif args.model == 'pointnetpp':

		model = PointNetPlusPlus(num_classes=args.num_classes)

	else:

		raise ValueError(f"Model {args.model} not supported")

	model = model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

	class_counts = [0] * args.num_classes

	print("Calculating weights for loss function...")

	for _, target in train_loader:

		valid_indices = target != -1

		if valid_indices.any():

			target_valid = target[valid_indices]

			for t in target_valid:

				class_counts[t.item()] += 1

	print(f"Class counts: {class_counts}")

	weights = torch.FloatTensor([1.0 / max(count, 1) for count in class_counts])
	weights = weights / weights.sum() * args.num_classes
	weights = weights.to(device)

	print(f"Weights for loss function: {weights}")

	criterion = nn.CrossEntropyLoss(weight=weights)

	best_acc = 0.0
	train_loss_history = []
	train_acc_history = []
	val_loss_history = []
	val_acc_history = []
	train_f1_history = []
	val_f1_history = []

	train_metrics = {
		'epoch': [],
		'loss': [],
		'accuracy': [],
		'precision_macro': [],
		'recall_macro': [],
		'f1_macro': [],
		'precision_weighted': [],
		'recall_weighted': [],
		'f1_weighted': [],
	}

	val_metrics = {
		'epoch': [],
		'loss': [],
		'accuracy': [],
		'precision_macro': [],
		'recall_macro': [],
		'f1_macro': [],
		'precision_weighted': [],
		'recall_weighted': [],
		'f1_weighted': [],
	}

	for epoch in range(args.epochs):

		model.train()
		train_loss = 0.0
		all_train_preds = []
		all_train_targets = []

		pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"epoch {epoch+1}/{args.epochs}")

		for i, (points, target) in pbar:

			valid_indices = target != -1

			if not valid_indices.any():

				continue

			points = points[valid_indices]
			target = target[valid_indices]

			points, target = points.to(device), target.to(device)
			points = points.transpose(2, 1)  # [B, N, C] -> [B, C, N]

			optimizer.zero_grad()

			if args.model == 'pointnet':

				pred, trans_feat = model(points)

			else:

				pred = model(points)
				trans_feat = None

			loss = criterion(pred, target)

			if args.feature_transform and args.model == 'pointnet':

				loss += args.feature_transform_regularizer * feature_transform_regularizer(trans_feat)

			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			_, predicted = torch.max(pred.data, 1)

			all_train_preds.extend(predicted.cpu().numpy())
			all_train_targets.extend(target.cpu().numpy())

			if len(all_train_targets) > 0:

				current_acc = accuracy_score(all_train_targets, all_train_preds)

				pbar.set_postfix({
					'loss': f"{train_loss/(i+1):.4f}",
					'acc': f"{current_acc:.4f}"
				})

			gc.collect()
			if torch.cuda.is_available():

				torch.cuda.empty_cache()

		if len(all_train_targets) > 0:

			train_loss = train_loss / len(train_loader)
			train_acc = accuracy_score(all_train_targets, all_train_preds)
			train_loss_history.append(train_loss)
			train_acc_history.append(train_acc)

			train_f1 = f1_score(all_train_targets, all_train_preds, average='macro')
			train_f1_history.append(train_f1)

			train_precision_macro = precision_score(all_train_targets, all_train_preds, average='macro', zero_division=0)
			train_recall_macro = recall_score(all_train_targets, all_train_preds, average='macro', zero_division=0)
			train_precision_weighted = precision_score(all_train_targets, all_train_preds, average='weighted', zero_division=0)
			train_recall_weighted = recall_score(all_train_targets, all_train_preds, average='weighted', zero_division=0)
			train_f1_weighted = f1_score(all_train_targets, all_train_preds, average='weighted', zero_division=0)

			train_metrics['epoch'].append(epoch + 1)
			train_metrics['loss'].append(train_loss)
			train_metrics['accuracy'].append(train_acc)
			train_metrics['precision_macro'].append(train_precision_macro)
			train_metrics['recall_macro'].append(train_recall_macro)
			train_metrics['f1_macro'].append(train_f1)
			train_metrics['precision_weighted'].append(train_precision_weighted)
			train_metrics['recall_weighted'].append(train_recall_weighted)
			train_metrics['f1_weighted'].append(train_f1_weighted)

		else:

			print("Warning: No valid labels in batch")

			train_loss_history.append(0)
			train_acc_history.append(0)
			train_f1 = 0
			train_f1_history.append(0)

		scheduler.step()

		model.eval()
		val_loss = 0.0
		all_val_preds = []
		all_val_targets = []

		with torch.no_grad():

			for points, target in val_loader:

				valid_indices = target != -1

				if not valid_indices.any():

					continue

				points = points[valid_indices]
				target = target[valid_indices]

				points, target = points.to(device), target.to(device)
				points = points.transpose(2, 1)  # [B, N, C] -> [B, C, N]

				if args.model == 'pointnet':

					pred, trans_feat = model(points)

				else:

					pred = model(points)

				loss = criterion(pred, target)
				val_loss += loss.item()

				_, predicted = torch.max(pred.data, 1)

				all_val_preds.extend(predicted.cpu().numpy())
				all_val_targets.extend(target.cpu().numpy())

				gc.collect()
				if torch.cuda.is_available():

					torch.cuda.empty_cache()

		if len(all_val_targets) > 0:

			val_loss = val_loss / len(val_loader)
			val_acc = accuracy_score(all_val_targets, all_val_preds)
			val_f1 = f1_score(all_val_targets, all_val_preds, average='macro')

			val_loss_history.append(val_loss)
			val_acc_history.append(val_acc)
			val_f1_history.append(val_f1)

			val_precision_macro = precision_score(all_val_targets, all_val_preds, average='macro', zero_division=0)
			val_recall_macro = recall_score(all_val_targets, all_val_preds, average='macro', zero_division=0)
			val_precision_weighted = precision_score(all_val_targets, all_val_preds, average='weighted', zero_division=0)
			val_recall_weighted = recall_score(all_val_targets, all_val_preds, average='weighted', zero_division=0)
			val_f1_weighted = f1_score(all_val_targets, all_val_preds, average='weighted', zero_division=0)

			val_metrics['epoch'].append(epoch + 1)
			val_metrics['loss'].append(val_loss)
			val_metrics['accuracy'].append(val_acc)
			val_metrics['precision_macro'].append(val_precision_macro)
			val_metrics['recall_macro'].append(val_recall_macro)
			val_metrics['f1_macro'].append(val_f1)
			val_metrics['precision_weighted'].append(val_precision_weighted)
			val_metrics['recall_weighted'].append(val_recall_weighted)
			val_metrics['f1_weighted'].append(val_f1_weighted)

			print(f"Training: Loss={train_loss:.4f}, Accuracy={train_acc:.4f}, F1={train_f1:.4f}")
			print(f"Validation: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}, F1={val_f1:.4f}")

			if val_f1 > best_acc:

				best_acc = val_f1

				checkpoint = {
					'epoch': epoch + 1,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'best_acc': val_acc,
					'best_f1': best_acc
				}

				model_dir = os.path.join(args.output_dir, f"{args.model}_best.pth")
				torch.save(checkpoint, model_dir)

				print(f"New best model saved with F1-score: {best_acc:.4f} | Model saved at: {model_dir}")

		if (epoch + 1) % args.checkpoint_interval == 0:

			checkpoint = {
				'epoch': epoch + 1,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'best_acc': best_acc
			}

			torch.save(checkpoint, os.path.join(args.output_dir, f"{args.model}_epoch_{epoch+1}.pth"))

	train_csv_path = os.path.join(args.output_dir, f"{args.model}_train_metrics.csv")
	val_csv_path = os.path.join(args.output_dir, f"{args.model}_val_metrics.csv")

	save_metrics_csv(train_metrics, train_csv_path)
	save_metrics_csv(val_metrics, val_csv_path)

	print(f"Training completed | Best F1-score: {best_acc:.4f}")

def inference(args):

	"""Perform inference on the KITTI dataset using a trained model"""

	device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

	if args.model == 'pointnet':

		model = PointNet(num_classes=args.num_classes, feature_transform=args.feature_transform)

	elif args.model == 'pointnetpp':

		model = PointNetPlusPlus(num_classes=args.num_classes)

	else:

		raise ValueError(f"Model {args.model} not supported")

	checkpoint = torch.load(args.model_path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'], strict=False)
	model = model.to(device)
	model.eval()

	print(f"Model loaded from {args.model_path}")
	print(f"Model accuracy: {checkpoint['best_acc']:.4f}")

	_, _, test_loader = get_kitti_object_dataloaders(
		root_dir=args.data_path,
		batch_size=args.batch_size,
		num_points=args.num_points
	)

	classes = ['Car', 'Pedestrian', 'Cyclist']

	all_preds = []
	all_targets = []

	memory_usages = []
	inference_times = []

	with torch.no_grad():

		for points, target in tqdm(test_loader, desc="Inference"):

			valid_indices = target != -1

			if not valid_indices.any():

				continue

			points = points[valid_indices]
			target = target[valid_indices]

			points, target = points.to(device), target.to(device)
			points = points.transpose(2, 1)  # [B, N, C] -> [B, C, N]

			inference_start = time.time()
			tracemalloc.start()

			if args.model == 'pointnet':

				pred, _ = model(points)

			else:

				pred = model(points)

			_, peak_memory = tracemalloc.get_traced_memory()
			inference_end = time.time()
			tracemalloc.stop()

			inference_time = inference_end - inference_start
			memory_usage = peak_memory / 10 ** 6

			inference_times.append(inference_time)
			memory_usages.append(memory_usage)

			_, predicted = torch.max(pred.data, 1)

			all_preds.extend(predicted.cpu().numpy())
			all_targets.extend(target.cpu().numpy())

			gc.collect()
			if torch.cuda.is_available():

				torch.cuda.empty_cache()

	all_preds = np.array(all_preds)
	all_targets = np.array(all_targets)

	accuracy = accuracy_score(all_targets, all_preds)
	print(f"\nGlobal accuracy: {accuracy:.4f}")

	cm = confusion_matrix(all_targets, all_preds)

	plt.figure(figsize=(10, 8))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, linewidths=0.5, linecolor='gray', cbar=True, square=True)

	plt.title('Confusion matrix', fontsize=16)
	plt.xlabel('Predicted label', fontsize=12)
	plt.ylabel('True label', fontsize=12)
	plt.xticks(rotation=45, ha='right')
	plt.yticks(rotation=0)
	plt.tight_layout()

	cbar = plt.gca().collections[0].colorbar
	cbar.set_label('Number of samples', labelpad=15)

	plt.savefig(os.path.join(args.output_dir, f"{args.model}_confusion_matrix.png"))

	print("\nDetailed metrics by class:")
	print(classification_report(all_targets, all_preds, target_names=classes, digits=4, zero_division=0))

	precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
	recall = recall_score(all_targets, all_preds, average=None, zero_division=0)
	f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

	print("\nMetrics by class:")
	for i in range(args.num_classes):
		print(f"{classes[i]}:")
		print(f"Precision: {precision[i]:.4f}")
		print(f"Recall: {recall[i]:.4f}")
		print(f"F1-Score: {f1[i]:.4f}")

	macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
	macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
	macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

	print("\nGeneral metrics (macro):")
	print(f"Precision: {macro_precision:.4f}")
	print(f"Recall: {macro_recall:.4f}")
	print(f"F1-Score: {macro_f1:.4f}")
	print(f"Memory usage (MB): {np.mean(memory_usages):.4f} ± {np.std(memory_usages):.4f}")
	print(f"Inference time (s): {np.mean(inference_times):.4f} ± {np.std(inference_times):.4f}")

	weighted_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
	weighted_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
	weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

	print("\nGeneral metrics (weighted):")
	print(f"Precision: {weighted_precision:.4f}")
	print(f"Recall: {weighted_recall:.4f}")
	print(f"F1-Score: {weighted_f1:.4f}")

	test_metrics = {
		'accuracy': [accuracy],
		'precision_macro': [macro_precision],
		'recall_macro': [macro_recall],
		'f1_macro': [macro_f1],
		'precision_weighted': [weighted_precision],
		'recall_weighted': [weighted_recall],
		'f1_weighted': [weighted_f1],
		'memory_usage': np.mean(memory_usages),
		'memory_usage_std': np.std(memory_usages),
		'memory_usage_single': np.mean(memory_usages),
		'memory_usage_single_std': np.std(memory_usages),
		'inference_time': np.mean(inference_times),
		'inference_time_std': np.std(inference_times),
		'inference_time_single': np.mean(inference_times),
		'inference_time_single_std': np.std(inference_times),
		'num_samples': len(all_targets),
		'num_classes': args.num_classes,
	}

	for i, cls in enumerate(classes):

		test_metrics[f'{cls}_precision'] = [precision[i]]
		test_metrics[f'{cls}_recall'] = [recall[i]]
		test_metrics[f'{cls}_f1'] = [f1[i]]

	test_csv_path = os.path.join(args.output_dir, f"{args.model}_test_metrics.csv")
	save_metrics_csv(test_metrics, test_csv_path)

	print(f"Inference completed | Results saved to {test_csv_path}")

def visualize_sample(args):
	"""Visualize a point cloud sample with its prediction"""

	device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

	if args.model == 'pointnet':

		model = PointNet(num_classes=args.num_classes, feature_transform=args.feature_transform)

	elif args.model == 'pointnetpp':

		model = PointNetPlusPlus(num_classes=args.num_classes)

	else:

		raise ValueError(f"Model {args.model} not supported")

	checkpoint = torch.load(args.model_path, map_location=device)

	model.load_state_dict(checkpoint['model_state_dict'], strict=False)

	model = model.to(device)

	model.eval()

	_, _, test_loader = get_kitti_object_dataloaders(
		root_dir=args.data_path,
		batch_size=1,
		num_points=args.num_points,
	)

	classes = ['Car', 'Pedestrian', 'Cyclist']

	found_valid_sample = False
	for points, target in test_loader:

		if target.item() != -1:

			found_valid_sample = True
			break

	if not found_valid_sample:

		print("No valid samples found in the test set.")

		return

	with torch.no_grad():

		points_cuda = points.to(device)
		points_cuda = points_cuda.transpose(2, 1)

		if args.model == 'pointnet':

			pred, _ = model(points_cuda)

		else:

			pred = model(points_cuda)

		_, predicted = torch.max(pred.data, 1)
		probs = torch.nn.functional.softmax(pred, dim=1)
		conf_score = probs[0][predicted.item()].item()

	points_np = points.squeeze().numpy()

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points_np)

	colors = np.zeros((points_np.shape[0], 3))

	if predicted.item() == 0:

		colors[:, 0] = 1.0  # Red for Car

	elif predicted.item() == 1:

		colors[:, 1] = 1.0  # Green for Pedestrian

	else:

		colors[:, 2] = 1.0  # Blue for Cyclist

	pcd.colors = o3d.utility.Vector3dVector(colors)

	true_label = "Unknown" if target.item() == -1 else classes[target.item()]
	pred_label = classes[predicted.item()]

	print(f"True label: {true_label}")
	print(f"Predicted label: {pred_label} (confidence: {conf_score:.4f})")

	print("\nProbabilities by class:")
	for i, cls in enumerate(classes):

		print(f"{cls}: {probs[0][i].item():.4f}")

	viz_metrics = {
		'confidence_score': [conf_score],
		'true_label': [true_label],
		'predicted_label': [pred_label]
	}

	for i, cls in enumerate(classes):
		viz_metrics[f'{cls}_probability'] = [probs[0][i].item()]

	viz_csv_path = os.path.join(args.output_dir, f"{args.model}_visualization_metrics.csv")
	save_metrics_csv(viz_metrics, viz_csv_path)

	o3d.visualization.draw_geometries([pcd], window_name=f"KITTI - Prediction: {pred_label} ({conf_score:.4f})", width=800, height=600)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='PointNet/PointNet++ for KITTI classification')

	parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'visualize'], help='Execution mode: train or infer')
	parser.add_argument('--data_path', type=str, required=True, help='Path to KITTI dataset')
	parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for models and results')
	parser.add_argument('--model', type=str, default='pointnet', choices=['pointnet', 'pointnetpp'], help='Model to use: pointnet or pointnetpp')
	parser.add_argument('--num_classes', type=int, default=3, help='Number of classes (default 3: Car, Pedestrian, Cyclist)')
	parser.add_argument('--num_points', type=int, default=4096, help='Number of points per cloud')
	parser.add_argument('--use_cuda', action='store_true', help='Enable CUDA')

	parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
	parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
	parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
	parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for L2 regularization')
	parser.add_argument('--feature_transform', action='store_true', help='Use feature transformation for PointNet')
	parser.add_argument('--feature_transform_regularizer', type=float, default=0.001, help='Weight for transformation matrix regularization')
	parser.add_argument('--checkpoint_interval', type=int, default=10, help='Epoch interval for saving checkpoints')

	parser.add_argument('--model_path', type=str, help='Path to trained model for inference/visualization')

	args = parser.parse_args()

	if not os.path.exists(args.output_dir):

		os.makedirs(args.output_dir)

	if args.mode == 'train':

		train(args)

	elif args.mode == 'inference':

		if args.model_path is None:

			parser.error("--model_path is required for 'inference' mode")

		inference(args)

	elif args.mode == 'visualize':

		if args.model_path is None:

			parser.error("--model_path is required for 'visualize' mode")

		visualize_sample(args)

	else:

		parser.error("Mode not supported. Use 'train', 'inference' or 'visualize'")