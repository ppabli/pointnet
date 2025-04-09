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
from sklearn.metrics import confusion_matrix

from pointnet import PointNet, feature_transform_regularizer
from pointnet_pp import PointNetPlusPlus
from kitti_dataset import get_kitti_object_dataloaders

def train(args):

	# Configuracion de dispositivo
	device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
	print(f"Usando dispositivo: {device}")

	# Dataloader con WeightedRandomSampler para manejar el desbalance durante el muestreo
	train_loader, val_loader, test_loader = get_kitti_object_dataloaders(
		root_dir=args.data_path,
		batch_size=args.batch_size,
		num_points=args.num_points
	)

	# Modelo
	if args.model == 'pointnet':

		model = PointNet(num_classes=args.num_classes, feature_transform=args.feature_transform)

	elif args.model == 'pointnetpp':

		model = PointNetPlusPlus(num_classes=args.num_classes)

	else:

		raise ValueError(f"Modelo {args.model} no soportado")

	model = model.to(device)

	# Optimizador y planificador
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

	# Tambien usamos pesos en la funcion de perdida para balance adicional
	# Este enfoque dual (WeightedRandomSampler + weighted loss) puede ser muy efectivo
	class_counts = [0] * args.num_classes
	print("Calculando pesos para la funcion de perdida...")

	# En vez de contar en el dataloader (ya balanceado por el sampler),
	# obtenemos los conteos de clases originales
	# Esto se puede extraer de train_dataset.data si lo exponemos
	for _, target in train_loader:

		valid_indices = target != -1

		if valid_indices.any():

			target_valid = target[valid_indices]

			for t in target_valid:

				class_counts[t.item()] += 1

	print(f"Conteo de clases: {class_counts}")

	# Cálculo de pesos inversos a la frecuencia
	weights = torch.FloatTensor([1.0/max(count, 1) for count in class_counts])
	weights = weights / weights.sum() * args.num_classes
	weights = weights.to(device)

	print(f"Pesos para las clases en la funcion de perdida: {weights}")

	# Funcion de perdida con pesos
	criterion = nn.CrossEntropyLoss(weight=weights)

	# Metricas de seguimiento
	best_acc = 0.0
	train_loss_history = []
	train_acc_history = []
	val_loss_history = []
	val_acc_history = []

	train_start = time.time()

	# Entrenamiento
	for epoch in range(args.epochs):

		# Entrenamiento
		model.train()
		train_loss = 0.0
		train_correct = 0
		train_total = 0

		pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"epoca {epoch+1}/{args.epochs}")

		for i, (points, target) in pbar:

			# Filtramos batches vacíos si aún quedan elementos con etiqueta -1
			valid_indices = target != -1

			if not valid_indices.any():

				continue

			# Si estamos usando el WeightedRandomSampler correctamente,
			# los batches ya deberían estar balanceados en terminos de clases
			# y contener menos elementos con etiqueta -1
			points = points[valid_indices]
			target = target[valid_indices]

			points, target = points.to(device), target.to(device)
			points = points.transpose(2, 1)  # [B, N, C] -> [B, C, N]

			optimizer.zero_grad()

			# Forward pass
			if args.model == 'pointnet':

				pred, trans_feat = model(points)

			else:

				pred = model(points)
				trans_feat = None

			# Cálculo de perdida
			loss = criterion(pred, target)

			if args.feature_transform and args.model == 'pointnet':
				loss += args.feature_transform_regularizer * feature_transform_regularizer(trans_feat)

			# Backward pass
			loss.backward()
			optimizer.step()

			# Estadísticas
			train_loss += loss.item()
			_, predicted = torch.max(pred.data, 1)
			train_total += target.size(0)
			train_correct += (predicted == target).sum().item()

			# Actualizar barra de progreso
			pbar.set_postfix({
				'loss': f"{train_loss/(i+1):.4f}",
				'acc': f"{100.0*train_correct/train_total:.2f}%"
			})

		# Calculamos metricas del epoch
		if train_total > 0:

			train_loss = train_loss / len(train_loader)
			train_acc = 100.0 * train_correct / train_total
			train_loss_history.append(train_loss)
			train_acc_history.append(train_acc)

		else:

			print("Warning: No valid training samples in this epoch")
			train_loss_history.append(0)
			train_acc_history.append(0)

		# Actualizar learning rate
		scheduler.step()

		# Evaluacion
		model.eval()
		val_loss = 0.0
		val_correct = 0
		val_total = 0

		with torch.no_grad():

			for points, target in val_loader:

				valid_indices = target != -1
				if not valid_indices.any():

					continue

				points = points[valid_indices]
				target = target[valid_indices]

				points, target = points.to(device), target.to(device)
				points = points.transpose(2, 1)  # [B, N, C] -> [B, C, N]

				# Forward pass
				if args.model == 'pointnet':

					pred, trans_feat = model(points)

				else:

					pred = model(points)

				# Cálculo de perdida
				loss = criterion(pred, target)
				val_loss += loss.item()

				# Estadísticas
				_, predicted = torch.max(pred.data, 1)
				val_total += target.size(0)
				val_correct += (predicted == target).sum().item()

		# Metricas de evaluacion
		if val_total > 0:

			val_loss = val_loss / len(val_loader)
			val_acc = 100.0 * val_correct / val_total
			val_loss_history.append(val_loss)
			val_acc_history.append(val_acc)

			print(f"epoca {epoch+1}/{args.epochs}")
			print(f"Entrenamiento: Perdida={train_loss:.4f}, Precision={train_acc:.2f}%")
			print(f"Evaluacion: Perdida={val_loss:.4f}, Precision={val_acc:.2f}%")

			# Guardar mejor modelo
			if val_acc > best_acc:

				best_acc = val_acc

				checkpoint = {
					'epoch': epoch + 1,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'best_acc': best_acc
				}

				model_dir = os.path.join(args.output_dir, f"{args.model}_best.pth")
				torch.save(checkpoint, model_dir)
				print(f"Nuevo mejor modelo guardado con precision: {best_acc:.2f}% | Modelo guardado en: {model_dir}")

		# Guardar checkpoint
		if (epoch + 1) % args.checkpoint_interval == 0:

			checkpoint = {
				'epoch': epoch + 1,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'best_acc': best_acc
			}

			torch.save(checkpoint, os.path.join(args.output_dir, f"{args.model}_epoch_{epoch+1}.pth"))

	train_end = time.time()

	print(f"Entrenamiento completado | Tiempo consumido: {train_end - train_start:.2f} segundos | Mejor precision: {best_acc:.2f}%")

def inference(args):

	# Configuracion de dispositivo
	device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

	# Cargar modelo
	if args.model == 'pointnet':

		model = PointNet(num_classes=args.num_classes, feature_transform=args.feature_transform)

	elif args.model == 'pointnetpp':

		model = PointNetPlusPlus(num_classes=args.num_classes)

	else:

		raise ValueError(f"Modelo {args.model} no soportado")

	checkpoint = torch.load(args.model_path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'], strict=False)
	model = model.to(device)
	model.eval()

	print(f"Modelo cargado desde {args.model_path}")
	print(f"Precision del modelo: {checkpoint['best_acc']:.2f}%")

	# Dataloader para inferencia
	_, _, test_loader = get_kitti_object_dataloaders(
		root_dir=args.data_path,
		batch_size=args.batch_size,
		num_points=args.num_points
	)

	# Clases
	classes = ['Car', 'Pedestrian', 'Cyclist']

	# Metricas
	confusion_matrix = np.zeros((args.num_classes, args.num_classes), dtype=int)
	class_correct = [0] * args.num_classes
	class_total = [0] * args.num_classes

	inference_start = time.time()

	with torch.no_grad():

		for points, target in tqdm(test_loader, desc="Inferencia"):

			# Ignoramos las muestras sin etiqueta (-1)
			valid_indices = target != -1

			if not valid_indices.any():

				continue

			points = points[valid_indices]
			target = target[valid_indices]

			points, target = points.to(device), target.to(device)
			points = points.transpose(2, 1)  # [B, N, C] -> [B, C, N]

			# Forward pass
			if args.model == 'pointnet':

				pred, _ = model(points)

			else:

				pred = model(points)

			# Predicciones
			_, predicted = torch.max(pred.data, 1)

			# Estadísticas por clase
			for i in range(target.size(0)):

				label = target[i].item()
				pred_label = predicted[i].item()

				confusion_matrix[label][pred_label] += 1
				class_total[label] += 1

				if label == pred_label:

					class_correct[label] += 1

	# Precision por clase
	print("\nPrecision por clase:")
	for i in range(args.num_classes):

		accuracy = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
		print(f"{classes[i]}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")

	# Matriz de confusion
	print("\nMatriz de confusion:")
	print("Verdadero\\Predicho", end="\t")
	for cls in classes:
		print(f"{cls}", end="\t")
	print()

	# Imprimir matriz de confusion


	for i in range(args.num_classes):

		print(f"{classes[i]}", end="\t\t")

		for j in range(args.num_classes):

			print(f"{confusion_matrix[i][j]}", end="\t")

		print()

	# Calcular precision, recall y F1-score por clase
	precision = np.zeros(args.num_classes)
	recall = np.zeros(args.num_classes)
	f1_score = np.zeros(args.num_classes)

	for i in range(args.num_classes):

		# Precision: TP / (TP + FP)
		# True positives: diagonal de la matriz de confusion
		true_positives = confusion_matrix[i][i]

		# False positives: suma de la columna i menos el valor en la diagonal
		false_positives = np.sum(confusion_matrix[:, i]) - true_positives

		# Precision
		precision[i] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

		# Recall: TP / (TP + FN)
		false_negatives = np.sum(confusion_matrix[i, :]) - true_positives

		# Recall
		recall[i] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

		# F1 Score: 2 * (precision * recall) / (precision + recall)
		f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

	# Imprimir resultados
	print("\nMetricas por clase:")

	for i in range(args.num_classes):

		print(f"{classes[i]}:")
		print(f"Precision: {precision[i]:.4f}")
		print(f"Recall: {recall[i]:.4f}")
		print(f"F1-Score: {f1_score[i]:.4f}")

	macro_precision = np.mean(precision)
	macro_recall = np.mean(recall)
	macro_f1 = np.mean(f1_score)

	print("\nMetricas generales (macro):")
	print(f"Precision: {macro_precision:.4f}")
	print(f"Recall: {macro_recall:.4f}")
	print(f"F1-Score: {macro_f1:.4f}")

	inference_end = time.time()

	print("Inferencia completada | Tiempo consumido: {:.2f} segundos".format(inference_end - inference_start))

def visualize_sample(args):
	"""Visualiza una muestra de nube de puntos con su prediccion"""

	# Configuracion
	device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

	# Cargar modelo
	if args.model == 'pointnet':

		model = PointNet(num_classes=args.num_classes, feature_transform=args.feature_transform)

	elif args.model == 'pointnetpp':

		model = PointNetPlusPlus(num_classes=args.num_classes)

	checkpoint = torch.load(args.model_path, map_location=device)

	model.load_state_dict(checkpoint['model_state_dict'], strict=False)

	model = model.to(device)

	model.eval()

	# Dataloader para una sola muestra
	_, _, test_loader = get_kitti_object_dataloaders(
		root_dir=args.data_path,
		batch_size=1,
		num_points=args.num_points
	)

	# Clases
	classes = ['Car', 'Pedestrian', 'Cyclist']

	# Obtener una muestra

	for points, target in test_loader:

		if target.item() != -1:

			break

	# Inferencia
	with torch.no_grad():

		points_cuda = points.to(device)
		points_cuda = points_cuda.transpose(2, 1)

		if args.model == 'pointnet':

			pred, _ = model(points_cuda)

		else:

			pred = model(points_cuda)

		_, predicted = torch.max(pred.data, 1)

	# Visualizacion
	points_np = points.squeeze().numpy()

	# Crear nube de puntos para Open3D
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points_np)

	# Colorear según la clase predicha
	colors = np.zeros((points_np.shape[0], 3))
	if predicted.item() == 0:  # Car - rojo

		colors[:, 0] = 1.0

	elif predicted.item() == 1:  # Pedestrian - verde

		colors[:, 1] = 1.0

	else:  # Cyclist - azul

		colors[:, 2] = 1.0

	pcd.colors = o3d.utility.Vector3dVector(colors)

	# Mostrar informacion
	true_label = "Desconocido" if target.item() == -1 else classes[target.item()]
	pred_label = classes[predicted.item()]
	print(f"Etiqueta verdadera: {true_label}")
	print(f"Etiqueta predicha: {pred_label}")

	# Visualizar
	o3d.visualization.draw_geometries([pcd], window_name=f"KITTI - Prediccion: {pred_label}", width=800, height=600)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='PointNet/PointNet++ para clasificacion KITTI')

	# Argumentos generales
	parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'visualize'], help='Modo de ejecucion: entrenar o inferir')
	parser.add_argument('--data_path', type=str, required=True, help='Ruta al dataset KITTI')
	parser.add_argument('--output_dir', type=str, default='./output', help='Directorio de salida para modelos y resultados')
	parser.add_argument('--model', type=str, default='pointnet', choices=['pointnet', 'pointnetpp'], help='Modelo a utilizar: pointnet o pointnetpp')
	parser.add_argument('--num_classes', type=int, default=3, help='Número de clases (por defecto 3: Car, Pedestrian, Cyclist)')
	parser.add_argument('--num_points', type=int, default=4096, help='Número de puntos por nube')
	parser.add_argument('--use_cuda', action='store_true', help='Deshabilitar CUDA')

	# Argumentos para entrenamiento
	parser.add_argument('--batch_size', type=int, default=32, help='Tamaño de batch para entrenamiento')
	parser.add_argument('--epochs', type=int, default=50, help='Número de epocas de entrenamiento')
	parser.add_argument('--lr', type=float, default=0.001, help='Tasa de aprendizaje inicial')
	parser.add_argument('--weight_decay', type=float, default=1e-4, help='Peso de decaimiento para regularizacion L2')
	parser.add_argument('--feature_transform', action='store_true', help='Usar transformacion de características para PointNet')
	parser.add_argument('--feature_transform_regularizer', type=float, default=0.001, help='Peso para regularizacion de matriz de transformacion')
	parser.add_argument('--checkpoint_interval', type=int, default=10, help='Intervalo de epocas para guardar checkpoints')

	# Argumentos para inferencia y visualizacion
	parser.add_argument('--model_path', type=str, help='Ruta al modelo entrenado para inferencia/visualizacion')

	args = parser.parse_args()

	# Crear directorio de salida si no existe
	if not os.path.exists(args.output_dir):

		os.makedirs(args.output_dir)

	# Ejecutar el modo seleccionado
	if args.mode == 'train':

		train(args)

	elif args.mode == 'inference':

		if args.model_path is None:

			parser.error("--model_path es requerido para el modo 'inference'")

		inference(args)

	elif args.mode == 'visualize':

		if args.model_path is None:

			parser.error("--model_path es requerido para el modo 'inference'")

		visualize_sample(args)

	else:

		parser.error("Modo no soportado. Usa 'train' o 'inference'")