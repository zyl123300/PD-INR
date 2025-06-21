import os
import numpy as np
import torch
import pickle
from data.data_utils import SystemMatrixGenerator, add_poisson_noise_tof, pet_forward
from src.utils import bi_load, bi_save4

system_matrix = SystemMatrixGenerator(
    path='./mMR_Geometry.mac',
    object_shape=(128, 128, 96),
    object_voxel_size=(1.5, 1.5, 1.5),
    gpu_index=1,
    TOF =True,
    n_subsets=14,
)()
pet = bi_load('./pet2_lesion_gt.bin',(128,128,96))
pet = torch.tensor(pet)
sino = pet_forward(system_matrix, pet, n_subsets=14)
sino[sino<0] = 0.0
print(sino.shape, sino.max(), sino.min(),sino.mean())
bi_save4(sino.cpu().numpy(),"./tof_sino_nnoise.bin")

sino_noise = add_poisson_noise_tof(sino).cpu()
sino_noise[sino_noise < 0] = 0.0
bi_save4(sino_noise.cpu().numpy(),"./tof_sino_noise.bin")

data = dict()
data["image"] = pet
data["sinogram"] = sino_noise
data["origin"] = np.array([0, 0, 0])
data["spacing"] = np.array([1.5, 1.5, 1.5])
data["shape"] = np.array([128, 128, 96])
data["macro_path"] = os.path.join('./mMR_Geometry.mac')
with open(os.path.join('./tof_pet_noise.pickle'), "wb") as handle:
    pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)
