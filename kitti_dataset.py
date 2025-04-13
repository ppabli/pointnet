import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import time
from torch.utils.data.sampler import WeightedRandomSampler

class KittiObjectDataset(Dataset):

	def __init__(self, root_dir, split='train', num_points=1024, transform=None, random_state=42, cache_dir=None):
		"""
		Dataset for classification of individual objects from the KITTI dataset

		Args:
			root_dir: Root directory where the KITTI dataset is located
			split: 'train', 'val', 'test'
			num_points: Number of points to use per object
			transform: Transformations to apply to the data
			random_state: Random seed for reproducibility
			cache_dir: Directory to cache processed objects
		"""

		self.root_dir = root_dir
		self.split = split
		self.num_points = num_points
		self.transform = transform
		self.cache_dir = cache_dir if cache_dir else os.path.join(root_dir, 'cache')

		if not os.path.exists(self.cache_dir):

			os.makedirs(self.cache_dir)

		self.cache_file = os.path.join(self.cache_dir, f'kitti_objects_{random_state}.pkl')

		self.classes = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'DontCare': -1}

		if os.path.exists(self.cache_file):

			print(f"Loading objects from cache: {self.cache_file}")
			self._load_cached_data()

		else:

			print("Processing KITTI data and creating cache...")
			self._process_kitti_data(random_state)

		if split == 'train':

			self.data = self.train_data

		elif split == 'val':

			self.data = self.val_data

		elif split == 'test':

			self.data = self.test_data

		print(f"Total objects in {split} split: {len(self.data)}")

		self._print_class_stats()

	def _process_kitti_data(self, random_state):
		"""Process KITTI data once and store the results"""

		self.data_dir = 'training'
		lidar_paths = glob.glob(os.path.join(self.root_dir, self.data_dir, 'velodyne', '*.bin'))
		label_paths = glob.glob(os.path.join(self.root_dir, self.data_dir, 'label_2', '*.txt'))
		calib_paths = glob.glob(os.path.join(self.root_dir, self.data_dir, 'calib', '*.txt'))

		lidar_paths.sort()
		label_paths.sort()
		calib_paths.sort()

		object_data = []

		print("Preprocessing KITTI data for object extraction...")

		tqdm_bar = tqdm(total=len(lidar_paths), desc="Processing scenes")

		for i in range(len(lidar_paths)):

			objects = self._extract_objects_from_scene(i, lidar_paths, label_paths, calib_paths)
			object_data.extend(objects)
			tqdm_bar.update(1)

		tqdm_bar.close()

		object_data = [obj for obj in object_data if obj['class_id'] != -1]

		train_val_data, self.test_data = train_test_split(
			object_data,
			test_size=0.2,
			train_size=0.8,
			random_state=random_state,
			stratify=[obj['class_id'] for obj in object_data]
		)

		self.train_data, self.val_data = train_test_split(
			train_val_data,
			test_size=0.25,
			train_size=0.75,
			random_state=random_state,
			stratify=[obj['class_id'] for obj in train_val_data]
		)

		self._save_cached_data()

	def _save_cached_data(self):
		"""Save processed data to cache"""

		cache_data = {
			'train_data': self.train_data,
			'val_data': self.val_data,
			'test_data': self.test_data,
			'timestamp': time.time()
		}

		print(f"Saving processed objects to cache: {self.cache_file}")

		with open(self.cache_file, 'wb') as f:

			pickle.dump(cache_data, f)

	def _load_cached_data(self):
		"""Load processed data from cache"""

		with open(self.cache_file, 'rb') as f:

			cache_data = pickle.load(f)

		self.train_data = cache_data['train_data']
		self.val_data = cache_data['val_data']
		self.test_data = cache_data['test_data']

		if 'timestamp' in cache_data:

			cache_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cache_data['timestamp']))

			print(f"Cache created at: {cache_time}")

	def _print_class_stats(self):
		"""Show class statistics in the dataset"""

		class_counts = {}

		for obj in self.data:

			class_id = obj['class_id']
			class_name = list(self.classes.keys())[list(self.classes.values()).index(class_id)]

			if class_name not in class_counts:

				class_counts[class_name] = 0

			class_counts[class_name] += 1

		print(f"Class distribution in {self.split}:")

		for class_name, count in class_counts.items():

			print(f"{class_name}: {count}")

	def _load_calib(self, calib_file):
		"""
		Load KITTI calibration data
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

		P2 = calib_data['P2'].reshape(3, 4)
		R0_rect = np.eye(4)
		R0_rect[:3, :3] = calib_data['R0_rect'].reshape(3, 3)
		Tr_velo_to_cam = np.eye(4)
		Tr_velo_to_cam[:3, :] = calib_data['Tr_velo_to_cam'].reshape(3, 4)

		return {'P2': P2, 'R0_rect': R0_rect, 'Tr_velo_to_cam': Tr_velo_to_cam}

	def _get_3d_box(self, h, w, l, x, y, z, rotation_y):
		"""
		Builds a 3D bounding box in camera coordinates

		Args:
			h, w, l: Height, width and length of the box
			x, y, z: Coordinates of the box center in camera coords
			rotation_y: Rotation around Y axis in camera coords

		Returns:
			corners_3d: Corners of the 3D box (8x3) in camera coords
		"""

		x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
		y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
		z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

		R = np.array([
			[np.cos(rotation_y), 0, np.sin(rotation_y)],
			[0, 1, 0],
			[-np.sin(rotation_y), 0, np.cos(rotation_y)]
		])

		corners_3d = np.zeros((8, 3))

		for i in range(8):

			corner = np.dot(R, np.array([x_corners[i], y_corners[i], z_corners[i]]))
			corners_3d[i] = corner + np.array([x, y, z])

		return corners_3d

	def _cam_to_velo(self, points_3d_cam, calib):
		"""
		Transforms points from camera coordinates to LiDAR coordinates

		Args:
			points_3d_cam: Points in camera coordinates (Nx3)
			calib: Dictionary with calibration matrices

		Returns:
			points_3d_velo: Points in LiDAR coordinates (Nx3)
		"""

		points_3d_cam_hom = np.hstack((points_3d_cam, np.ones((points_3d_cam.shape[0], 1))))

		R0_rect_inv = np.linalg.inv(calib['R0_rect'])
		points_3d_cam_nonrect = np.dot(points_3d_cam_hom, R0_rect_inv.T)

		Tr_cam_to_velo = np.linalg.inv(calib['Tr_velo_to_cam'])
		points_3d_velo = np.dot(points_3d_cam_nonrect, Tr_cam_to_velo.T)

		return points_3d_velo[:, :3]

	def _extract_points_in_box(self, points, box_corners_velo):
		"""
		Extracts points inside a 3D bounding box in LiDAR coordinates

		Args:
			points: Point cloud in LiDAR coords (Nx3)
			box_corners_velo: 3D box corners in LiDAR coords (8x3)

		Returns:
			points_in_box: Points inside the 3D box
		"""

		box_center = np.mean(box_corners_velo, axis=0)

		front_bottom_right = box_corners_velo[0]
		front_bottom_left = box_corners_velo[1]
		back_bottom_left = box_corners_velo[2]
		front_top_right = box_corners_velo[4]

		v_longitudinal = front_bottom_left - back_bottom_left
		v_longitudinal = v_longitudinal / np.linalg.norm(v_longitudinal)

		v_vertical = front_top_right - front_bottom_right
		v_vertical = v_vertical / np.linalg.norm(v_vertical)

		v_lateral = front_bottom_right - front_bottom_left
		v_lateral = v_lateral / np.linalg.norm(v_lateral)

		length = np.linalg.norm(front_bottom_left - back_bottom_left)
		height = np.linalg.norm(front_top_right - front_bottom_right)
		width = np.linalg.norm(front_bottom_right - front_bottom_left)

		points_centered = points - box_center

		proj_longitudinal = np.dot(points_centered, v_longitudinal)
		proj_vertical = np.dot(points_centered, v_vertical)
		proj_lateral = np.dot(points_centered, v_lateral)

		in_longitudinal = np.logical_and(proj_longitudinal >= -length/2, proj_longitudinal <= length/2)
		in_vertical = np.logical_and(proj_vertical >= -height/2, proj_vertical <= height/2)
		in_lateral = np.logical_and(proj_lateral >= -width/2, proj_lateral <= width/2)

		mask = np.logical_and(np.logical_and(in_longitudinal, in_vertical), in_lateral)

		return points[mask]

	def _extract_objects_from_scene(self, idx, lidar_paths, label_paths, calib_paths, min_points=50):
		"""Extract individual objects from a KITTI scene"""

		lidar_file = lidar_paths[idx]
		point_cloud = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
		points_xyz = point_cloud[:, :3]

		label_file = label_paths[idx]
		with open(label_file, 'r') as f:

			objects = f.readlines()

		calib_file = calib_paths[idx]
		calib = self._load_calib(calib_file)

		extracted_objects = []

		for obj in objects:

			parts = obj.strip().split(' ')
			obj_type = parts[0]

			if obj_type not in self.classes:

				continue

			class_id = self.classes[obj_type]

			if class_id == -1:

				continue

			h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
			x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
			rotation_y = float(parts[14])

			box_3d_cam = self._get_3d_box(h, w, l, x, y, z, rotation_y)

			box_3d_velo = self._cam_to_velo(box_3d_cam, calib)

			object_points = self._extract_points_in_box(points_xyz, box_3d_velo)

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

		if len(points) >= self.num_points:

			indices = np.random.choice(len(points), self.num_points, replace=False)
			points = points[indices]

		else:

			indices = np.random.choice(len(points), self.num_points, replace=True)
			points = points[indices]

		points = points - np.mean(points, axis=0)

		if np.max(np.abs(points)) > 0:
			points = points / np.max(np.abs(points))

		if self.transform:

			points = self.transform(points)

		return torch.from_numpy(points).float(), torch.tensor(class_id, dtype=torch.long)

class PointCloudRotation:

	def __call__(self, points):

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
	Create dataloaders for KITTI object classification

	Args:
		root_dir: Root directory where the KITTI dataset is located
		batch_size: Batch size
		num_points: Number of points per object
		cache_dir: Directory to cache processed objects

	Returns:
		train_loader, val_loader, test_loader: DataLoaders for training, validation and testing
	"""

	train_transforms = [
		PointCloudRotation(),
		PointCloudScale(),
		PointCloudJitter()
	]

	def train_transform(points):

		for t in train_transforms:

			points = t(points)

		return points

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
		transform=None,  # No augmentation in validation
		cache_dir=cache_dir
	)

	test_dataset = KittiObjectDataset(
		root_dir=root_dir,
		split='test',
		num_points=num_points,
		transform=None,  # No transformations for test
		cache_dir=cache_dir
	)

	class_counts = {}
	for obj in train_dataset.data:

		class_id = obj['class_id']

		if class_id not in class_counts:

			class_counts[class_id] = 0

		class_counts[class_id] += 1

	class_weights = {}
	num_samples = len(train_dataset.data)
	num_classes = len(class_counts)

	for class_id, count in class_counts.items():

		class_weights[class_id] = num_samples / (num_classes * count)

	sample_weights = []
	for obj in train_dataset.data:

		class_id = obj['class_id']
		weight = class_weights[class_id]
		sample_weights.append(weight)

	sampler = WeightedRandomSampler(
		weights=sample_weights,
		num_samples=len(sample_weights),
		replacement=True
	)

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
		shuffle=True,
		num_workers=4
	)

	return train_loader, val_loader, test_loader