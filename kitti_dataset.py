import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import time
import matplotlib.pyplot as plt
from torch.utils.data.sampler import WeightedRandomSampler

class KittiObjectDataset(Dataset):

	def __init__(self, root_dir, split='train', num_points=1024, transform=None, random_state=42, cache_dir=None):
		"""
		Dataset para clasificacion de objetos individuales del dataset KITTI

		Args:
			root_dir: Directorio raiz donde se encuentra el dataset KITTI
			split: 'train', 'val', 'test'
			num_points: Número de puntos a usar por objeto
			transform: Transformaciones a aplicar a los datos
			random_state: Semilla aleatoria para reproducibilidad
			cache_dir: Directorio para cachear los objetos procesados
		"""

		self.root_dir = root_dir
		self.split = split
		self.num_points = num_points
		self.transform = transform
		self.cache_dir = cache_dir if cache_dir else os.path.join(root_dir, 'cache')

		# Crear directorio de cache si no existe
		if not os.path.exists(self.cache_dir):

			os.makedirs(self.cache_dir)

		# Nombre del archivo de cache
		self.cache_file = os.path.join(self.cache_dir, f'kitti_objects_{random_state}.pkl')

		# Clases a detectar
		self.classes = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'DontCare': -1}

		# Comprobamos si existe el archivo de cache y lo cargamos
		# En caso contrario, procesamos los datos
		if os.path.exists(self.cache_file):

			print(f"Cargando objetos desde cache: {self.cache_file}")
			self._load_cached_data()

		else:

			print("Procesando datos KITTI y creando cache...")
			self._process_kitti_data(random_state)

		# Asignar según el split
		if split == 'train':
			self.data = self.train_data
		elif split == 'val':
			self.data = self.val_data
		elif split == 'test':
			self.data = self.test_data

		print(f"Total objetos en split {split}: {len(self.data)}")

		# Mostramos estadisticas de clases
		self._print_class_stats()

	def _process_kitti_data(self, random_state):
		"""Procesa los datos KITTI una sola vez y almacena los resultados"""

		# Usamos el directorio training de KITTI para train/val
		self.data_dir = 'training'
		lidar_paths = glob.glob(os.path.join(self.root_dir, self.data_dir, 'velodyne', '*.bin'))
		label_paths = glob.glob(os.path.join(self.root_dir, self.data_dir, 'label_2', '*.txt'))
		calib_paths = glob.glob(os.path.join(self.root_dir, self.data_dir, 'calib', '*.txt'))

		# Ordenamos para asegurar correspondencia
		lidar_paths.sort()
		label_paths.sort()
		calib_paths.sort()

		# Preparamos lista para almacenar objetos
		object_data = []

		# Procesamos cada escena para extraer objetos individuales
		print("Preprocesando datos KITTI para extraccion de objetos...")

		# Display progress bar
		tqdm_bar = tqdm(total=len(lidar_paths), desc="Procesando escenas")

		for i in range(len(lidar_paths)):

			# Procesamos cada escena y extraemos los objetos
			objects = self._extract_objects_from_scene(i, lidar_paths, label_paths, calib_paths)
			object_data.extend(objects)
			tqdm_bar.update(1)

		tqdm_bar.close()

		# Filtramos objetos no validos (DontCare o sin clase conocida)
		object_data = [obj for obj in object_data if obj['class_id'] != -1]

		# Dividimos en train, val y test
		train_val_data, self.test_data = train_test_split(
			object_data,
			test_size=0.2,
			train_size=0.8,
			random_state=random_state,
			stratify=[obj['class_id'] for obj in object_data]
		)

		# Luego dividir train_val en train y validation
		# 25% de 80% = 20% del total para validacion
		self.train_data, self.val_data = train_test_split(
			train_val_data,
			test_size=0.25,
			train_size=0.75,
			random_state=random_state,
			stratify=[obj['class_id'] for obj in train_val_data]
		)

		# Guardamos los datos procesados para futuras ejecuciones
		self._save_cached_data()

	def _save_cached_data(self):
		"""Guarda los datos procesados en cache"""

		cache_data = {
			'train_data': self.train_data,
			'val_data': self.val_data,
			'test_data': self.test_data,
			'timestamp': time.time()
		}

		print(f"Guardando objetos procesados en cache: {self.cache_file}")

		with open(self.cache_file, 'wb') as f:

			pickle.dump(cache_data, f)

	def _load_cached_data(self):
		"""Carga los datos procesados desde cache"""

		with open(self.cache_file, 'rb') as f:

			cache_data = pickle.load(f)

		self.train_data = cache_data['train_data']
		self.val_data = cache_data['val_data']
		self.test_data = cache_data['test_data']

		# Mostrar informacion de cuando se creo la cache
		if 'timestamp' in cache_data:

			cache_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cache_data['timestamp']))

			print(f"Cache creada en: {cache_time}")

	def _print_class_stats(self):
		"""Muestra estadisticas de clases en el conjunto de datos"""

		if self.split == 'test':

			print(f"Conjunto de test: {len(self.data)} objetos")
			return

		class_counts = {}

		for obj in self.data:

			class_id = obj['class_id']

			class_name = list(self.classes.keys())[list(self.classes.values()).index(class_id)]
			if class_name not in class_counts:

				class_counts[class_name] = 0

			class_counts[class_name] += 1

		print(f"Distribucion de clases en {self.split}:")

		for class_name, count in class_counts.items():

			print(f"{class_name}: {count}")

	def _load_calib(self, calib_file):
		"""
		Carga los datos de calibracion KITTI
		"""

		with open(calib_file, 'r') as f:

			lines = f.readlines()

		calib_data = {}
		for line in lines:

			line = line.strip()

			if not line:

				continue

			key, value = line.split(':', 1)
			calib_data[key] = np.array([float(x) for x in value.split()])

		# Matrices de calibracion
		P2 = calib_data['P2'].reshape(3, 4)  # Camara 2 matriz de proyeccion
		R0_rect = np.eye(4)  # Matriz de rectificacion
		R0_rect[:3, :3] = calib_data['R0_rect'].reshape(3, 3)
		Tr_velo_to_cam = np.eye(4)  # Matriz de transformacion del LiDAR a la camara
		Tr_velo_to_cam[:3, :] = calib_data['Tr_velo_to_cam'].reshape(3, 4)

		return {'P2': P2, 'R0_rect': R0_rect, 'Tr_velo_to_cam': Tr_velo_to_cam}

	def _get_3d_box(self, h, w, l, x, y, z, rotation_y):
		"""
		Construye un cuadro delimitador 3D en coordenadas de camara

		Args:
			h, w, l: Altura, anchura y longitud del cuadro
			x, y, z: Coordenadas del centro del cuadro en coords de camara
			rotation_y: Rotacion alrededor del eje Y en coords de camara

		Returns:
			corners_3d: Esquinas del cuadro 3D (8x3) en coords de camara
		"""

		# Crear cuadro 3D centrado en el origen y alineado con los ejes
		# En KITTI: x = right, y = down, z = forward (en coords de camara)
		x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
		y_corners = [0, 0, 0, 0, -h, -h, -h, -h]  # El origen esta en el suelo
		z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

		# Rotacion alrededor del eje Y
		R = np.array([
			[np.cos(rotation_y), 0, np.sin(rotation_y)],
			[0, 1, 0],
			[-np.sin(rotation_y), 0, np.cos(rotation_y)]
		])

		# Obtener las coordenadas de cada esquina en coords de camara
		corners_3d = np.zeros((8, 3))

		for i in range(8):

			corner = np.dot(R, np.array([x_corners[i], y_corners[i], z_corners[i]]))
			corners_3d[i] = corner + np.array([x, y, z])

		return corners_3d

	def _cam_to_velo(self, points_3d_cam, calib):
		"""
		Transforma puntos de coordenadas de camara a coordenadas de LiDAR

		Args:
			points_3d_cam: Puntos en coordenadas de camara (Nx3)
			calib: Diccionario con matrices de calibracion

		Returns:
			points_3d_velo: Puntos en coordenadas de LiDAR (Nx3)
		"""

		# Convertir puntos a formato homogeneo
		points_3d_cam_hom = np.hstack((points_3d_cam, np.ones((points_3d_cam.shape[0], 1))))

		# Transformacion de camara rectificada a camara no rectificada
		R0_rect_inv = np.linalg.inv(calib['R0_rect'])
		points_3d_cam_nonrect = np.dot(points_3d_cam_hom, R0_rect_inv.T)

		# Transformacion de camara a LiDAR (inversa de Tr_velo_to_cam)
		Tr_cam_to_velo = np.linalg.inv(calib['Tr_velo_to_cam'])
		points_3d_velo = np.dot(points_3d_cam_nonrect, Tr_cam_to_velo.T)

		# Eliminar coordenada homogenea
		return points_3d_velo[:, :3]

	def _extract_points_in_box(self, points, box_corners_velo):
		"""
		Extrae puntos dentro de un cuadro delimitador 3D en coordenadas de LiDAR

		Args:
			points: Nube de puntos en coords de LiDAR (Nx3)
			box_corners_velo: Esquinas del cuadro 3D en coords de LiDAR (8x3)

		Returns:
			points_in_box: Puntos dentro del cuadro 3D
		"""

		# Calculamos los vectores de los ejes del box
		box_center = np.mean(box_corners_velo, axis=0)

		# Ordenamos las esquinas para obtener los ejes correctamente
		# (esto depende del orden exacto de las esquinas en _get_3d_box)
		front_bottom_right = box_corners_velo[0]
		front_bottom_left = box_corners_velo[1]
		back_bottom_left = box_corners_velo[2]
		front_top_right = box_corners_velo[4]

		# Calculamos los ejes del box (en coords de LiDAR despues de la transformacion)
		v_longitudinal = front_bottom_left - back_bottom_left  # eje hacia adelante
		v_longitudinal = v_longitudinal / np.linalg.norm(v_longitudinal)

		v_vertical = front_top_right - front_bottom_right  # eje hacia arriba
		v_vertical = v_vertical / np.linalg.norm(v_vertical)

		v_lateral = front_bottom_right - front_bottom_left  # eje hacia el lado
		v_lateral = v_lateral / np.linalg.norm(v_lateral)

		# Calculamos dimensiones del box
		length = np.linalg.norm(front_bottom_left - back_bottom_left)
		height = np.linalg.norm(front_top_right - front_bottom_right)
		width = np.linalg.norm(front_bottom_right - front_bottom_left)

		# Centramos los puntos
		points_centered = points - box_center

		# Proyectamos los puntos a los ejes del box
		proj_longitudinal = np.dot(points_centered, v_longitudinal)
		proj_vertical = np.dot(points_centered, v_vertical)
		proj_lateral = np.dot(points_centered, v_lateral)

		# Verificamos si los puntos estan dentro del box
		in_longitudinal = np.logical_and(proj_longitudinal >= -length/2, proj_longitudinal <= length/2)
		in_vertical = np.logical_and(proj_vertical >= -height/2, proj_vertical <= height/2)
		in_lateral = np.logical_and(proj_lateral >= -width/2, proj_lateral <= width/2)

		# Combinamos las condiciones
		mask = np.logical_and(np.logical_and(in_longitudinal, in_vertical), in_lateral)

		# Devolvemos los puntos dentro del box
		return points[mask]

	def _extract_objects_from_scene(self, idx, lidar_paths, label_paths, calib_paths, min_points=50):
		"""Extrae objetos individuales de una escena KITTI"""

		# Cargamos la nube de puntos
		lidar_file = lidar_paths[idx]
		point_cloud = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
		points_xyz = point_cloud[:, :3]

		# Cargamos las etiquetas
		label_file = label_paths[idx]
		with open(label_file, 'r') as f:

			objects = f.readlines()

		# Cargamos los datos de calibracion
		calib_file = calib_paths[idx]
		calib = self._load_calib(calib_file)

		# Extraemos objetos individuales
		extracted_objects = []

		for obj in objects:

			parts = obj.strip().split(' ')
			obj_type = parts[0]  # 'Car', 'Pedestrian', etc.

			# Solo procesamos clases conocidas, Car, Pedestrian y Cyclist
			if obj_type not in self.classes:

				continue

			# Obtenemos el ID de clase
			class_id = self.classes[obj_type]

			# Ignoramos objetos marcados como DontCare
			if class_id == -1:

				continue

			# Extraemos parametros del cuadro delimitador 3D
			h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
			x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
			rotation_y = float(parts[14])

			# Construimos cuadro delimitador 3D en coordenadas de camara
			box_3d_cam = self._get_3d_box(h, w, l, x, y, z, rotation_y)

			# Transformamos el box 3D de coordenadas de camara a coordenadas de LiDAR
			box_3d_velo = self._cam_to_velo(box_3d_cam, calib)

			# Filtramos puntos dentro del cuadro 3D
			object_points = self._extract_points_in_box(points_xyz, box_3d_velo)

			# Solo guardamos objetos con suficientes puntos
			if len(object_points) >= min_points:

				extracted_objects.append({
					'points': object_points,
					'class_id': class_id,
					'dimensions': [h, w, l],
					'location': [x, y, z],
					'rotation_y': rotation_y,
					'scene_idx': idx
				})

		return extracted_objects

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):

		obj_data = self.data[idx]
		points = obj_data['points']
		class_id = obj_data['class_id']

		# Realizamos subsampling para obtener num_points
		if len(points) >= self.num_points:

			indices = np.random.choice(len(points), self.num_points, replace=False)
			points = points[indices]

		else:

			# Repetimos puntos si hay menos que num_points
			indices = np.random.choice(len(points), self.num_points, replace=True)
			points = points[indices]

		# Centramos y normalizamos los puntos
		points = points - np.mean(points, axis=0)

		# Escalamos a un rango unitario
		if np.max(np.abs(points)) > 0:

			points = points / np.max(np.abs(points))

		# Aplicamos transformaciones si existen
		if self.transform:

			points = self.transform(points)

		return torch.from_numpy(points).float(), torch.tensor(class_id, dtype=torch.long)

# Aumentacion de datos para nubes de puntos
class PointCloudRotation:

	def __call__(self, points):

		# Rotacion aleatoria alrededor del eje z
		theta = np.random.uniform(0, 2 * np.pi)

		rotation_matrix = np.array([
			[np.cos(theta), -np.sin(theta), 0],
			[np.sin(theta), np.cos(theta), 0],
			[0, 0, 1]
		])

		return np.dot(points, rotation_matrix)

class PointCloudScale:

	def __call__(self, points):

		scale = np.random.uniform(0.8, 1.2)
		return points * scale

class PointCloudJitter:

	def __init__(self, sigma=0.01, clip=0.05):

		self.sigma = sigma
		self.clip = clip

	def __call__(self, points):

		jitter = np.clip(self.sigma * np.random.randn(*points.shape), -self.clip, self.clip)
		return points + jitter

def get_kitti_object_dataloaders(root_dir, batch_size=32, num_points=1024, cache_dir=None):
	"""
	Crea dataloaders para clasificacion de objetos KITTI

	Args:
		root_dir: Directorio raiz donde se encuentra el dataset KITTI
		batch_size: Tamaño del batch
		num_points: Número de puntos por objeto
		cache_dir: Directorio para cachear los objetos procesados

	Returns:
		train_loader, val_loader, test_loader: DataLoaders para entrenamiento, validacion y test
	"""

	# Transformaciones para aumentacion de datos
	train_transforms = [
		PointCloudRotation(),
		PointCloudScale(),
		PointCloudJitter()
	]

	def train_transform(points):
		for t in train_transforms:
			points = t(points)
		return points

	# Datasets
	train_dataset = KittiObjectDataset(
		root_dir=root_dir,
		split='train',
		num_points=num_points,
		transform=train_transform,
		cache_dir=cache_dir
	)

	val_dataset = KittiObjectDataset(
		root_dir=root_dir,
		split='val',
		num_points=num_points,
		transform=None,  # No aplicamos aumentacion en validacion
		cache_dir=cache_dir
	)

	test_dataset = KittiObjectDataset(
		root_dir=root_dir,
		split='test',
		num_points=num_points,
		transform=None,  # Sin transformaciones para test
		cache_dir=cache_dir
	)

	# Calculamos los pesos para el WeightedRandomSampler
	# Contamos las instancias de cada clase
	class_counts = {}
	for obj in train_dataset.data:
		class_id = obj['class_id']
		if class_id not in class_counts:
			class_counts[class_id] = 0
		class_counts[class_id] += 1

	# Calculamos los pesos inversos para cada clase
	class_weights = {}
	num_samples = len(train_dataset.data)
	num_classes = len(class_counts)

	for class_id, count in class_counts.items():
		# Invertimos la frecuencia para dar mas peso a clases menos representadas
		class_weights[class_id] = num_samples / (num_classes * count)

	# Asignamos un peso a cada muestra en el dataset
	sample_weights = []

	for obj in train_dataset.data:

		class_id = obj['class_id']
		weight = class_weights[class_id]
		sample_weights.append(weight)

	# Creamos el WeightedRandomSampler
	sampler = WeightedRandomSampler(
		weights=sample_weights,
		num_samples=len(sample_weights),
		replacement=True
	)

	# DataLoaders
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=4,
		drop_last=True,
		sampler=sampler
	)

	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=4
	)

	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=4
	)

	return train_loader, val_loader, test_loader