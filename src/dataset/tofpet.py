import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pytomography.metadata import ObjectMeta
from data.data_utils import  SystemMatrixGenerator

class TOFPETDataset(Dataset):
    def __init__(self, path, n_rays=1024, device="cuda"):
        super().__init__()

        with open(path, "rb") as handle:
            data = pickle.load(handle)
        if "image" in data.keys():
            self.image = data["image"].to(device)
        else:
            self.image = None
        # Specify object space for reconstruction
        object_meta = ObjectMeta(
            dr=data["spacing"],  # mm
            shape=data["shape"]  # voxels (128,128,96)
        )
        system_matrix = SystemMatrixGenerator(
            path='data/mMR_Geometry.mac',
            object_shape=data["shape"],
            object_voxel_size=data["spacing"],
            TOF=True,
            gpu_index=1,
        )()
        self.system_matrix = system_matrix
        self.tof_meta = system_matrix.proj_meta.tof_meta

        # self.type = type
        self.n_rays = n_rays

        # rays = self.get_rays()
        xyz1, xyz2 = system_matrix._get_xyz_sinogram_coordinates(subset_idx=None)
        directions = xyz2 - xyz1
        rays = torch.cat([xyz1, directions], dim=-1) / 1000 # mm -> m
        sinogram = data["sinogram"].to(torch.device('cuda:2')).div_(1000)
        shape = (*sinogram.shape[:-1], 3)
        self.rays = torch.cat([xyz1.reshape(shape), directions.reshape(shape)], dim=-1) / 1000
        self.nVoxel = np.array(object_meta.shape)
        self.dVoxel = np.array(system_matrix.object_meta.dr) / 1000
        self.origin = data["origin"]  # (- np.array(object_meta.shape) / 2 + 0.5) * (np.array(self.object_dr))
        self.sVoxel = self.nVoxel * self.dVoxel

        near, far = self.get_near_far_with_box(rays, "cpu")
        self.rays_concat = torch.cat([rays, near.unsqueeze(-1), far.unsqueeze(-1)], dim=-1)
        self.voxels = torch.tensor(self.get_voxels(), dtype=torch.float32, device=device)

        self.projs = sinogram
        self.projs_concat = self.projs.flatten(0, -2)  # (102989824,13)
        self.projs_nonTOF = torch.sum(sinogram, -1)  # (224,449,4096)
        self.projs_nonTOF_concat = self.projs_nonTOF.flatten()  # (102989824)
        projs_valid = (self.rays_concat[..., 7] > self.rays_concat[..., 6])  # & (self.projs_nonTOF_concat > 0)

        self.projs_concat = self.projs_concat[projs_valid]
        self.rays_concat = self.rays_concat[projs_valid]

        self.update_indices()
        torch.cuda.empty_cache()
        self.device = device

    def __len__(self):
        return self.indices.shape[0]

    def update_indices(self):
        indices = np.arange(self.rays_concat.shape[0])

        num = self.rays_concat.shape[0] // self.n_rays
        if num * self.n_rays < self.rays_concat.shape[0]:
            num_com = (num + 1) * self.n_rays - self.rays_concat.shape[0]

            indices_com = np.random.choice(self.rays_concat.shape[0], size=[num_com], replace=False)
            indices = np.concatenate((indices, indices_com))

        np.random.shuffle(indices)

        self.indices = np.reshape(indices, (int(indices.shape[0] / self.n_rays), self.n_rays))

    def update_indices_OS(self, num_set=10):
        pass

    def __getitem__(self, index):
        # if self.type == "train":
        select_inds = self.indices[index]
        self.select_inds = select_inds
        rays = self.rays_concat[select_inds].to(self.device)
        projs = self.projs_concat[select_inds].to(self.device)
        # print(projs.shape)
        # print(rays.shape)
        out = {
            "projs": projs,
            "rays": rays,
        }

        # elif self.type == "val":
        #     rays = self.rays[index].to(self.device)
        #     projs = self.projs[index].to(self.device)
        #     out = {
        #         "projs":projs,
        #         "rays":rays,
        #     }
        return out

    def get_voxels(self):
        """
        Get the voxels.
        """
        n1, n2, n3 = self.nVoxel
        o1, o2, o3 = self.origin
        s1, s2, s3 = self.sVoxel / 2 - self.dVoxel / 2

        xyz = np.meshgrid(np.linspace(-s1 + o1, s1 + o1, n1),
                          np.linspace(-s2 + o2, s2 + o2, n2),
                          np.linspace(-s3 + o3, s3 + o3, n3), indexing="ij")
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
        return voxel

    def get_near_far_with_box(self, rays, device):
        # please refer to https://gist.github.com/DomNomNom/46bb1ce47f68d255fd5d
        # rays = rays.cpu().detach().numpy()
        # s1, s2, s3 = self.sVoxel / 2 # + geo.dVoxel / 2

        box_min = -self.sVoxel / 2 + self.origin
        box_max = self.sVoxel / 2 + self.origin

        box_min = torch.tensor(box_min, dtype=torch.float32, device=device)
        box_max = torch.tensor(box_max, dtype=torch.float32, device=device)

        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]

        t_min = (box_min.cpu() - rays_o.cpu()) / rays_d.cpu()
        t_max = (box_max.cpu() - rays_o.cpu()) / rays_d.cpu()

        # t_min = t_min.to(device)
        # t_max = t_max.to(device)

        t1 = torch.minimum(t_min, t_max)
        t2 = torch.maximum(t_min, t_max)

        near = torch.max(t1, dim=-1)
        far = torch.min(t2, dim=-1)
        # check if far > near
        return near.values.to(device), far.values.to(device)


