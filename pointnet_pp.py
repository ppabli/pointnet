import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src, dst):
	"""
	Calcula la distancia euclidiana al cuadrado entre todos los puntos de src y dst.
	"""

	B, N, _ = src.shape
	_, M, _ = dst.shape
	dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
	dist += torch.sum(src ** 2, -1).view(B, N, 1)
	dist += torch.sum(dst ** 2, -1).view(B, 1, M)

	return dist

def index_points(points, idx):
	"""
	Indexa puntos según índices dados.
	"""

	device = points.device
	B = points.shape[0]
	view_shape = list(idx.shape)
	view_shape[1:] = [1] * (len(view_shape) - 1)
	repeat_shape = list(idx.shape)
	repeat_shape[0] = 1
	batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
	new_points = points[batch_indices, idx, :]

	return new_points

def farthest_point_sample(xyz, npoint):
	"""
	Muestreo de puntos más lejanos.
	"""

	device = xyz.device
	B, N, C = xyz.shape
	centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
	distance = torch.ones(B, N).to(device) * 1e10
	farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
	batch_indices = torch.arange(B, dtype=torch.long).to(device)

	for i in range(npoint):

		centroids[:, i] = farthest
		centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
		dist = torch.sum((xyz - centroid) ** 2, -1)
		mask = dist < distance
		distance[mask] = dist[mask]
		farthest = torch.max(distance, -1)[1]

	return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
	"""
	Encuentra puntos dentro de una esfera de radio dado.
	"""

	device = xyz.device
	B, N, C = xyz.shape
	_, S, _ = new_xyz.shape
	group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
	sqrdists = square_distance(new_xyz, xyz)
	group_idx[sqrdists > radius ** 2] = N
	group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
	group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
	mask = group_idx == N
	group_idx[mask] = group_first[mask]

	return group_idx

# Capa de Set Abstraction para PointNet++
class SetAbstraction(nn.Module):

	def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):

		super(SetAbstraction, self).__init__()

		self.npoint = npoint
		self.radius = radius
		self.nsample = nsample
		self.group_all = group_all
		self.mlp_convs = nn.ModuleList()
		self.mlp_bns = nn.ModuleList()

		last_channel = in_channel + 3

		for out_channel in mlp:

			self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
			self.mlp_bns.append(nn.BatchNorm2d(out_channel))
			last_channel = out_channel

	def forward(self, xyz, points):
		"""
		Input:
			xyz: coordenadas de los puntos [B, N, 3]
			points: características de los puntos [B, N, D]
		Return:
			new_xyz: coordenadas de los puntos muestreados [B, npoint, 3]
			new_points: características agrupadas [B, npoint, mlp[-1]]
		"""

		device = xyz.device
		B, N, C = xyz.shape

		if self.group_all:

			new_xyz = torch.zeros(B, 1, C).to(device)
			grouped_xyz = xyz.view(B, 1, N, C)

		else:

			fps_idx = farthest_point_sample(xyz, self.npoint)
			new_xyz = index_points(xyz, fps_idx)

			idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
			grouped_xyz = index_points(xyz, idx)

			grouped_xyz -= new_xyz.view(B, self.npoint, 1, C)

		if points is not None:

			if self.group_all:

				grouped_points = points.view(B, 1, N, -1)

			else:

				grouped_points = index_points(points, idx)

			grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)

		else:

			grouped_points = grouped_xyz

		grouped_points = grouped_points.permute(0, 3, 2, 1)

		for i, conv in enumerate(self.mlp_convs):

			grouped_points = F.relu(self.mlp_bns[i](conv(grouped_points)))

		new_points = torch.max(grouped_points, 2)[0]
		new_points = new_points.permute(0, 2, 1)

		return new_xyz, new_points

class PointNetPlusPlus(nn.Module):

	def __init__(self, num_classes=3, normal_channel=False):

		super(PointNetPlusPlus, self).__init__()
		self.normal_channel = normal_channel
		in_channel = 6 if normal_channel else 3

		self.sa1 = SetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
		self.sa2 = SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256], group_all=False)
		self.sa3 = SetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, mlp=[256, 512, 1024], group_all=True)

		self.fc1 = nn.Linear(1024, 512)
		self.bn1 = nn.BatchNorm1d(512)
		self.drop1 = nn.Dropout(0.4)
		self.fc2 = nn.Linear(512, 256)
		self.bn2 = nn.BatchNorm1d(256)
		self.drop2 = nn.Dropout(0.4)
		self.fc3 = nn.Linear(256, num_classes)

	def forward(self, xyz):

		B, _, _ = xyz.shape

		if self.normal_channel:

			norm = xyz[:, 3:, :]
			xyz = xyz[:, :3, :]

		else:

			norm = None

		# Capa 1 de Set Abstraction
		l1_xyz, l1_points = self.sa1(xyz, norm)

		# Capa 2 de Set Abstraction
		l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

		# Capa 3 de Set Abstraction (agrupamiento global)
		_, l3_points = self.sa3(l2_xyz, l2_points)

		# Flatten
		x = l3_points.view(B, 1024)

		# Capas fully-connected
		x = self.drop1(F.relu(self.bn1(self.fc1(x))))
		x = self.drop2(F.relu(self.bn2(self.fc2(x))))
		x = self.fc3(x)

		return x