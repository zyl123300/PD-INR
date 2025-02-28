import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tinycudann as tcnn

class HashGridEncoder(nn.Module):
    def __init__(self, input_dim, num_levels=16, level_dim=2,
                 base_resolution=16,
                 log2_hashmap_size=19, per_level_scale=2):
        super().__init__()

        encoding_config = {"otype": "Grid",
                           "type": "Hash",
                           "n_levels": num_levels,
                           "n_features_per_level": level_dim,
                           "log2_hashmap_size": log2_hashmap_size,
                           "base_resolution": base_resolution,
                           "per_level_scale": per_level_scale,
                           "interpolation": "Linear"}

        self.encoder = tcnn.Encoding(input_dim, encoding_config)
        self.output_dim = level_dim * num_levels

    def forward(self, x, aabb, bound=(-1, 1)):
        x = normalize_aabb(x, aabb)
        x = (x - bound[0]) / (bound[1] - bound[0])
        return self.encoder(x)  # .to(torch.float32)

def normalize_aabb(pts, aabb):
    if pts.shape[-1] <= 3:
        return pts / aabb
    else:
        pts[..., :3] = pts[..., :3] / aabb
        pts[..., 3] = 2 * pts[..., 3] - 1
        return pts
class ImagePriorEncoder(nn.Module):
    def __init__(self, image, voxels, sVoxel, voxels_shape = (128,128,96)):
        super().__init__()
        self.image = image
        self.output_dim = 1
        self.voxels = voxels
        self.sVoxel = sVoxel
        self.bound = torch.tensor(sVoxel / 2, device=self.voxels.device)
        self.features = None
        self.value_grid_sample()
        self.voxels_shape = voxels_shape

    def update_image_prior(self, image):
        voxels_shape = self.voxels_shape
        x = self.voxels.reshape(-1, 3)
        x = normalize_aabb(x, self.bound)
        bound = (-1, 1)
        x = (x - bound[0]) / (bound[1] - bound[0])
        x = x * 2 - 1
        x = x[:, [2, 1, 0]]
        device = x.device
        image = image.to(device).unsqueeze(0).unsqueeze(0).float()
        grid = x.view(1, -1, 1, 1, 3).float()
        center = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze()
        features = center.unsqueeze(1)
        features = features.view(*voxels_shape, -1)
        self.features = features

    def value_grid_sample(self):
        voxels_shape = self.voxels_shape
        x = self.voxels.reshape(-1, 3)
        x = normalize_aabb(x, self.bound)
        bound = (-1, 1)
        x = (x - bound[0]) / (bound[1] - bound[0])
        x = x * 2 - 1  # [-1, 1]
        x = x[:, [2, 1, 0]]
        device = x.device

        image = self.image.to(device).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 64, 64, 48)
        grid = x.view(1, -1, 1, 1, 3).float()

        center = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze()

        offsets = torch.tensor([
            [1, 0, 0], [-1, 0, 0],  # x 方向
            [0, 1, 0], [0, -1, 0],  # y 方向
            [0, 0, 1], [0, 0, -1],  # z 方向
        ], device=device, dtype=torch.float32)
        offsets = torch.tensor([
            [1, 0, 0], [-1, 0, 0],
            [2, 0, 0], [-2, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 2, 0], [0, -2, 0],
            [0, 0, 1], [0, 0, -1],
            [0, 0, 2], [0, 0, -2],
        ], device=device, dtype=torch.float32)

        neighbors = []
        for offset in offsets:
            neighbor_coords = x + offset / torch.tensor([64, 64, 48], device=device, dtype=torch.float32) * 2  # 偏移归一化
            grid = neighbor_coords.view(1, -1, 1, 1, 3).float()
            neighbor = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze()
            neighbors.append(neighbor)

        features = center.unsqueeze(1)
        self.features = features.view(*voxels_shape, -1)

        return self.features

    def feature_grid_sample(self, x):
        x = x * 2 - 1
        x = x[:, [2, 1, 0]]

        device = self.features.device
        features = self.features.permute(3, 0, 1, 2).unsqueeze(0).to(device)
        grid = x.view(1, -1, 1, 1, 3).to(device)
        sampled_features = F.grid_sample(features, grid, mode='nearest', padding_mode='border', align_corners=True)
        sampled_features = sampled_features.squeeze(3).squeeze(3).squeeze(0)  # 移除多余维度
        if sampled_features.dim() == 3:  # 形状为 (C, N, 1)
            sampled_features = sampled_features.squeeze(-1)  # 去掉最后的维度
        elif sampled_features.dim() != 2:
            raise RuntimeError(f"Unexpected shape of sampled_features: {sampled_features.shape}")

        sampled_features = sampled_features.permute(1, 0)  # (C, N) -> (N, C)
        return sampled_features


    def forward(self, x, aabb, bound=(-1, 1)):
        x = normalize_aabb(x, aabb)
        x = (x - bound[0]) / (bound[1] - bound[0])
        feature_vector = self.feature_grid_sample(x)
        return feature_vector

class ImagePriorHashEncoder(nn.Module):
    def __init__(self, voxels, sVoxel, input_dim, prior_image_path, voxels_shape=(128,128,96), num_levels=16, level_dim=2,
                 base_resolution=16,
                 log2_hashmap_size=19, per_level_scale=2, ):
        super().__init__()
        self.voxels = voxels
        self.sVoxel = sVoxel
        self.voxels_shape = voxels_shape
        self.hash_encoder = HashGridEncoder(input_dim, num_levels, level_dim, base_resolution, log2_hashmap_size,
                                            per_level_scale)

        shape = tuple(reversed(voxels_shape))
        data = (np.fromfile(prior_image_path, dtype='float32').reshape(shape).astype(np.float32))
        data = np.flip(np.transpose(data, (2, 1, 0)), axis=[0, 2]).copy()
        image = torch.from_numpy(data)
        # image = self.gaussian_blur_3d(image)
        image64 = self.downsample_3d(image, scale_factor=2)
        self.image_prior_encoder = ImagePriorEncoder(image64, self.voxels, self.sVoxel, self.voxels_shape)
        self.output_dim = self.image_prior_encoder.output_dim + self.hash_encoder.output_dim

    def update_image_prior(self, image):
        self.image_prior_encoder.update_image_prior(image)

    def downsample_3d(self, image, scale_factor):
        if scale_factor not in [2, 4]:
            raise ValueError("Scale factor must be 2 or 4.")
        image = image.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)

        downsampled_image = F.interpolate(
            image,
            scale_factor=1 / scale_factor,
            mode='trilinear',
            align_corners=False
        )

        return downsampled_image.squeeze(0).squeeze(0)

    def maxpooling_sample_3d(self, image, scale_factor):
        if scale_factor not in [2, 4]:
            raise ValueError("Scale factor must be 2 or 4.")
        image = image.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
        pool = nn.MaxPool3d(kernel_size=scale_factor, stride=scale_factor)
        downsampled_image = pool(image)

        return downsampled_image.squeeze(0).squeeze(0)  # 移除batch和channel维度

    def gaussian_blur_3d(self, input_tensor, kernel_size=5, sigma=1.0):
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        def gaussian_1d(size, sigma):
            x = torch.linspace(-size // 2, size // 2, steps=size)
            g = torch.exp(-0.5 * (x / sigma) ** 2)
            return g / g.sum()

        kernel_1d = gaussian_1d(kernel_size, sigma)
        kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
        kernel_3d = kernel_3d / kernel_3d.sum()  # Normalize the kernel
        kernel_3d = kernel_3d.to(input_tensor.device).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        blurred_tensor = torch.nn.functional.conv3d(
            input_tensor, kernel_3d, padding=kernel_size // 2
        )

        return blurred_tensor.squeeze(0).squeeze(0)

    def forward(self, x, aabb):
        hash_feature = self.hash_encoder(x, aabb)
        image_feature = self.image_prior_encoder(x, aabb)
        return torch.cat([hash_feature, image_feature], dim=-1)
