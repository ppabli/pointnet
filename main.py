import os
import argparse
import torch

from train import train, inference, visualize_sample

def main():

	parser = argparse.ArgumentParser(description='Sistema de clasificacion de nubes de puntos LiDAR usando KITTI')

	# Argumentos principales
	parser.add_argument('--action', type=str, required=True, choices=['train', 'test'], help='Accion a realizar')

	# Argumentos para rutas
	parser.add_argument('--data_dir', type=str, default='./data/kitti', help='Directorio del dataset KITTI')
	parser.add_argument('--output_dir', type=str, default='./output', help='Directorio para guardar resultados')

	# Argumentos para modelo
	parser.add_argument('--model', type=str, default='pointnet', choices=['pointnet', 'pointnetpp'], help='Modelo a utilizar')
	parser.add_argument('--model_path', type=str, default=None, help='Ruta al modelo pre-entrenado')

	# Argumentos para entrenamiento
	parser.add_argument('--epochs', type=int, default=50, help='Numero de epocas')
	parser.add_argument('--batch_size', type=int, default=32, help='Tamaño de batch')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Tasa de aprendizaje')
	parser.add_argument('--num_points', type=int, default=4096, help='Numero de puntos a utilizar por muestra')
	parser.add_argument('--feature_transform', action='store_true', help='Usar transformacion de características para PointNet')

	# Argumentos para visualizacion
	parser.add_argument('--sample_idx', type=int, default=0, help='Índice de la muestra a visualizar')

	args = parser.parse_args()

	# Aseguramos que el directorio de salida exista
	os.makedirs(args.output_dir, exist_ok=True)

	if args.action == 'train':

		print(f"Entrenando modelo {args.model} con datos de {args.data_dir}")

		train_args = argparse.Namespace(
			mode='train',
			data_path=args.data_dir,
			output_dir=args.output_dir,
			model=args.model,
			num_classes=3,
			num_points=args.num_points,
			no_cuda=not torch.cuda.is_available(),
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

			print("Error: Se requiere --model_path para testing")
			return

		print(f"Evaluando modelo {args.model} desde {args.model_path}")

		test_args = argparse.Namespace(
			mode='inference',
			data_path=args.data_dir,
			output_dir=args.output_dir,
			model=args.model,
			num_classes=3,
			num_points=args.num_points,
			use_cuda=not torch.cuda.is_available(),
			batch_size=args.batch_size,
			model_path=args.model_path,
			feature_transform=args.feature_transform
		)

		inference(test_args)

	elif args.action == 'visualize':

		if args.model_path is None:

			print("Error: Se requiere --model_path para visualizacion")
			return

		print(f"Visualizando resultados usando {args.model_path}")

		vis_args = argparse.Namespace(
			mode='visualize',
			data_path=args.data_dir,
			output_dir=args.output_dir,
			model=args.model,
			num_classes=3,
			num_points=args.num_points,
			use_cuda=not torch.cuda.is_available(),
			model_path=args.model_path,
			feature_transform=args.feature_transform
		)

		visualize_sample(vis_args)

	else:

		print("Accion no valida. Usa 'train', 'test' o 'visualize'")
		return


if __name__ == "__main__":

	main()