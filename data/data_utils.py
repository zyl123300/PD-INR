from pytomography.io.PET import gate
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETTOFMeta, PETSinogramPolygonProjMeta
import pytomography
from pytomography.projectors import SystemMatrix
from collections.abc import Callable
import torch
import numpy as np
import parallelproj


def add_poisson_noise_tof(sino_tof):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    shape = sino_tof.shape
    sino_tof = sino_tof.contiguous().view(-1, 13).to(device)
    n_rays = 16384
    num = (sino_tof.shape[0] + n_rays - 1) // n_rays
    for i in range(num):
        sino_tof_subset = sino_tof[i * n_rays:(i + 1) * n_rays, :]
        sino_ntof_subset = sino_tof_subset.sum(dim=-1, keepdim=True)
        scale = 0.1 / (sino_ntof_subset + 1e-10)
        sino_tof_subset.mul_(scale)
        sino_tof_subset.copy_(torch.poisson(sino_tof_subset))
        sino_tof_subset.div_(scale)
        sino_tof[i * n_rays:(i + 1) * n_rays, :] = sino_tof_subset
    sino_tof = sino_tof.view(shape)
    return sino_tof
def get_attenuation_map_bin(path:str,shape:tuple):
    umap=bi_load(path, shape).copy()
    return torch.tensor(umap)/10
def bi_load(path: str, shape: tuple):
    shape = tuple(reversed(shape))
    data = (np.fromfile(path, dtype='float32').reshape(shape).astype(np.float32))
    data=np.flip(np.transpose(data, (2, 1, 0)), axis=[0, 2]).copy()
    return data
def pet_forward(H, im, n_subsets = 24):
    H.set_n_subsets(n_subsets)
    sinogram = torch.zeros(H.proj_meta.shape).unsqueeze(-1).repeat(1,1,1,H.proj_meta.tof_meta.num_bins).to(pytomography.dtype).to(H.output_device)
    for subset_idx in range(n_subsets):
        sinogram[H.subset_indices_array[subset_idx]] = H.forward(im, subset_idx)
    return sinogram

class SystemMatrixGenerator:
    def __init__(self, path='./mMR_Geometry.mac',atten_path=None, gpu_index=0, num_tof_bins=13, fwhm_tof_resolution_ps=200,
                 object_shape=(128, 128, 96), object_voxel_size=(2, 2, 2),n_subsets=1,TOF = False):
        self.path = path
        self.atten_path = atten_path
        self.gpu_index = gpu_index
        # self.n_subsets = n_subsets
        self.device = torch.device(index=self.gpu_index, type='cuda')
        self.num_tof_bins = num_tof_bins
        self.fwhm_tof_resolution_ps = fwhm_tof_resolution_ps
        self.object_shape = object_shape
        self.object_voxel_size = object_voxel_size
        self.system_matrix = None
        self.n_subsets = n_subsets
        self.TOF=TOF

    def set_device(self):
        """Sets the device for pytomography operations."""
        pytomography.set_device(self.device)

    def get_scanner_info(self):
        """Loads the scanner geometry and information."""
        macro_path = self.path
        info = gate.get_detector_info(path=macro_path, mean_interaction_depth=9, min_rsector_difference=0)
        return info

    def create_tof_meta(self):
        """Creates TOF metadata based on provided parameters."""
        speed_of_light = 0.3  # mm/ps
        fwhm_tof_resolution = self.fwhm_tof_resolution_ps * speed_of_light / 2  # ps to position along LOR
        TOF_range = 1000 * speed_of_light  # ps to position along LOR (full range)
        tof_meta = PETTOFMeta(self.num_tof_bins, TOF_range, fwhm_tof_resolution, n_sigmas=3)
        self.tof_meata=tof_meta
        return tof_meta

    def create_object_meta(self):
        """Creates object metadata for PET reconstruction."""
        object_meta=ObjectMeta(
            dr=self.object_voxel_size,  # mm per voxel
            shape=self.object_shape  # voxel dimensions
        )
        self.object_meta = object_meta
        return object_meta
    def create_proj_meta(self):
        # Load scanner info
        info = self.get_scanner_info()
        if self.TOF:
            tof_meta = self.create_tof_meta()
            proj_meta = PETSinogramPolygonProjMeta(info, tof_meta=tof_meta)
        else:
            proj_meta = PETSinogramPolygonProjMeta(info)
        self.proj_meta = proj_meta
        return proj_meta
    def generate_system_matrix(self):
        """Generates the PET system matrix."""
        self.set_device()
        # Create TOF and object metadata
        object_meta = self.create_object_meta()
        # Create projection metadata
        proj_meta = self.create_proj_meta()
        # Generate the system matrix
        if self.atten_path is not None:
            # atten_map = gate.get_attenuation_map_nifti(self.atten_path, object_meta).to(pytomography.device)
            atten_map=get_attenuation_map_bin(self.atten_path,self.object_shape).to(pytomography.device)
        else:
            atten_map=None
        self.system_matrix = TOF_PETSystemMatrix(
            object_meta,
            proj_meta,
            sinogram_sensitivity=None,
            obj2obj_transforms=[],
            # scale_projection_by_sensitivity=True,
            attenuation_map=atten_map,
            N_splits=10,  # Split FP/BP into 10 loops to save memory
            device="cpu",  # projections are output on the CPU, but internal computation is on GPU
            n_subsets=self.n_subsets,
        )
        print("System matrix created successfully.")
        # return self.system_matrix
    def __call__(self, *args, **kwargs):
        self.generate_system_matrix()
        return self.system_matrix
class TOF_PETSystemMatrix(SystemMatrix):
    r"""System matrix for sinogram-based PET reconstruction.

        Args:
            object_meta (ObjectMeta): Metadata of object space, containing information on voxel size and dimensions.
            proj_meta (PETSinogramPolygonProjMeta): PET sinogram projection space metadata. This information contains the scanner lookup table and time-of-flight metadata.
            obj2obj_transforms (list[Transform], optional): Object to object space transforms applied before forward projection and after back projection. These are typically used for PSF modeling in PET imaging. Defaults to [].
            attenuation_map (torch.tensor | None, optional): Attenuation map used for attenuation modeling. If provided, all weights will be scaled by detection probabilities derived from this map. Note that this scales on top of ``sinogram_sensitivity``, so if attenuation is already accounted for there, this is not needed. Defaults to None.
            sinogram_sensitivity (torch.tensor | None, optional): Normalization sinogram used to scale projections after forward projection. This factor may include detector normalization :math:`\eta` and/or attenuation modeling :math:`\mu`. The attenuation modeling :math:`\mu` should not be included if ``attenuation_map`` is provided as an argument to the function. Defaults to None.
            scale_projection_by_sensitivity (bool, optional): Whether or not to scale the projections by :math:`\mu \eta`. This is not needed in reconstruction algorithms using a PoissonLogLikelihood. Defaults to False.
            N_splits (int, optional): Splits up computation of forward/back projection to save GPU memory. Defaults to 1.
            device (str, optional): The device for any objects in projection space projection space (what it outputs in forward projection and what it expects for back projection). This is seperate from ``pytomography.device`` since the internal functionality may still use GPU even if this is CPU. This is used to save GPU memory since sinograms are often very large. Defaults to pytomography.device.
        """

    def __init__(
            self,
            object_meta: ObjectMeta,
            proj_meta: PETSinogramPolygonProjMeta,
            obj2obj_transforms: [],
            attenuation_map: torch.Tensor | None = None,
            sinogram_sensitivity: torch.Tensor | None = None,
            scale_projection_by_sensitivity: bool = True,
            N_splits: int = 1,
            device: str = pytomography.device,
            n_subsets: int = 1,
    ) -> None:
        super(TOF_PETSystemMatrix, self).__init__(
            obj2obj_transforms=obj2obj_transforms,
            proj2proj_transforms=[],
            object_meta=object_meta,
            proj_meta=proj_meta
        )
        self.output_device = device
        self.object_origin = (- np.array(object_meta.shape) / 2 + 0.5) * (np.array(object_meta.dr))
        self.obj2obj_transforms = obj2obj_transforms
        self.proj_meta = proj_meta
        # In case they get put on another device
        self.attenuation_map = attenuation_map
        self.sinogram_sensitivity = sinogram_sensitivity
        self.scale_projection_by_sensitivity = scale_projection_by_sensitivity
        if sinogram_sensitivity is not None:
            self.sinogram_sensitivity = self.sinogram_sensitivity.to(self.output_device)
        self.N_splits = N_splits
        self.TOF = self.proj_meta.tof_meta is not None
        self.n_subsets = n_subsets
    def _get_xyz_sinogram_coordinates(self, subset_idx: int = None):
        """Get the XYZ coordinates corresponding to the pair of crystals of the projection angle

        Args:
            subset_idx (int, optional): Subset index for ths sinogram. If None, considers all elements. Defaults to None.

        Returns:
            Sequence[torch.Tensor, torch.Tensor]: XYZ coordinates of crystal 1 and XYZ coordinates of crystal 2 corresponding to all elements in the sinogram.
        """
        if subset_idx is not None:
            idx = self.subset_indices_array[subset_idx].cpu()
            detector_coordinates = self.proj_meta.detector_coordinates[idx]
            N_angles = idx.shape[0]
        else:
            detector_coordinates = self.proj_meta.detector_coordinates
            N_angles = self.proj_meta.shape[0]
        xy1 = torch.flatten(detector_coordinates, start_dim=0, end_dim=1)[:, 0].cpu()
        xy2 = torch.flatten(detector_coordinates, start_dim=0, end_dim=1)[:, 1].cpu()
        z1, z2 = self.proj_meta.ring_coordinates.T.cpu()
        xyz1 = torch.concatenate([
            xy1.unsqueeze(1).repeat(1, z1.shape[0], 1),
            z1.unsqueeze(0).unsqueeze(-1).repeat(xy1.shape[0], 1, 1)
        ], dim=-1).flatten(start_dim=0, end_dim=1)
        xyz2 = torch.concatenate([
            xy2.unsqueeze(1).repeat(1, z2.shape[0], 1),
            z2.unsqueeze(0).unsqueeze(-1).repeat(xy2.shape[0], 1, 1)
        ], dim=-1).flatten(start_dim=0, end_dim=1)
        xyz1 = xyz1.reshape((N_angles, *self.proj_meta.shape[1:], 3))
        xyz2 = xyz2.reshape((N_angles, *self.proj_meta.shape[1:], 3))
        return xyz1.flatten(start_dim=0, end_dim=2), xyz2.flatten(start_dim=0, end_dim=2)

    def _compute_atteunation_probability_projection(self, subset_idx: torch.tensor) -> torch.tensor:
        """Compute the probability of a photon not being attenuated for a certain sinogram element.

        Args:
            subset_idx (torch.tensor): Subset index for ths sinogram.

        Returns:
            torch.tensor: Probability sinogram
        """
        xyz1, xyz2 = self._get_xyz_sinogram_coordinates(subset_idx=subset_idx)
        proj = torch.zeros(xyz1.shape[0]).to(self.output_device)
        for idx_partial in torch.tensor_split(torch.arange(xyz1.shape[0]), self.N_splits):
            proj[idx_partial] += torch.exp(-parallelproj.joseph3d_fwd(
                xyz1[idx_partial].to(pytomography.device),
                xyz2[idx_partial].to(pytomography.device),
                self.attenuation_map.to(pytomography.device),
                self.object_origin,
                self.object_meta.dr
            )).to(self.output_device)
        N_angles = self.proj_meta.N_angles if subset_idx is None else self.subset_indices_array[subset_idx].shape[0]
        proj = proj.reshape((N_angles, *self.proj_meta.shape[1:]))
        return proj

    def _compute_sensitivity_sinogram(self, subset_idx: int = None):
        r"""Computes the sensitivity sinogram :math:`\mu \eta` that accounts for attenuation effects and normalization effects.

        Args:
            subset_idx (int, optional): Subset index for ths sinogram. If None, considers all elements. Defaults to None..

        Returns:
            torch.Tensor: Sensitivity sinogram.
        """
        if self.sinogram_sensitivity is not None:
            sinogram_sensitivity = self.sinogram_sensitivity
        else:
            sinogram_sensitivity = torch.ones(self.proj_meta.shape).to(self.output_device)
        if subset_idx is not None:
            sinogram_sensitivity = self.get_projection_subset(sinogram_sensitivity, subset_idx)
        # Scale the weights by attenuation image if its provided in the system matrix
        if self.attenuation_map is not None:
            sinogram_sensitivity = sinogram_sensitivity * self._compute_atteunation_probability_projection(subset_idx)
        if self.TOF:
            sinogram_sensitivity = sinogram_sensitivity.unsqueeze(-1)
        return sinogram_sensitivity

    def set_n_subsets(self, n_subsets: int) -> list:
        """Returns a list where each element consists of an array of indices corresponding to a partitioned version of the projections.

        Args:
            n_subsets (int): Number of subsets to partition the projections into

        Returns:
            list: List of arrays where each array corresponds to the projection indices of a particular subset.
        """
        indices = torch.arange(self.proj_meta.N_angles).to(torch.long).to(self.output_device)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        self.subset_indices_array = subset_indices_array

    def get_projection_subset(self, projections: torch.Tensor, subset_idx: int | None) -> torch.tensor:
        """Obtains subsampled projections :math:`g_m` corresponding to subset index :math:`m`. Sinogram PET partitions projections based on angle.

        Args:
            projections (torch.Tensor): total projections :math:`g`
            subset_idx (int): subset index :math:`m`

        Returns:
            torch.Tensor: subsampled projections :math:`g_m`.
        """
        if subset_idx is None:
            return projections
        else:
            subset_indices = self.subset_indices_array[subset_idx]
            proj_subset = projections[subset_indices]
            return proj_subset

    def get_weighting_subset(
            self,
            subset_idx: int
    ) -> float:
        r"""Computes the relative weighting of a given subset (given that the projection space is reduced). This is used for scaling parameters relative to :math:`\tilde{H}_m^T 1` in reconstruction algorithms, such as prior weighting :math:`\beta`

        Args:
            subset_idx (int): Subset index

        Returns:
            float: Weighting for the subset.
        """
        if subset_idx is None:
            return 1
        else:
            return len(self.subset_indices_array[subset_idx]) / self.proj_meta.N_angles

    def compute_normalization_factor(self, subset_idx: int = None):
        r"""Computes the normalization factor :math:`H^T \mu \eta`

        Args:
            subset_idx (int, optional): Subset index for ths sinogram. If None, considers all elements. Defaults to None..

        Returns:
            torch.Tensor: Normalization factor.
        """
        return self.backward(1, subset_idx, force_nonTOF=True, force_scale_by_sensitivity=True)

    def forward(
            self,
            object: torch.tensor,
            subset_idx: int = None,
    ) -> torch.tensor:
        r"""PET Sinogram forward projection

        Args:
            object (torch.tensor): Object to be forward projected
            subset_idx (int, optional): Subset index for ths sinogram. If None, considers all elements. Defaults to None.
            scale_by_sensitivity (bool, optional): Whether or not to scale the projections by :math:`\mu \eta`. This is not necessarily needed in reconstruction algorithms. Defaults to False.

        Returns:
            torch.tensor: Forward projection
        """
        # Apply object space transforms
        object = object.to(pytomography.device)
        for transform in self.obj2obj_transforms:
            object = transform.forward(object)
        # Project
        xyz1, xyz2 = self._get_xyz_sinogram_coordinates(subset_idx=subset_idx)
        if self.TOF:
            proj = torch.zeros((xyz1.shape[0], self.proj_meta.tof_meta.num_bins)).to(pytomography.dtype).to(
                self.output_device)
        else:
            proj = torch.zeros((xyz1.shape[0])).to(pytomography.dtype).to(self.output_device)
        for idx_partial in torch.tensor_split(torch.arange(xyz1.shape[0]), self.N_splits):
            if self.TOF:
                proj[idx_partial] += parallelproj.joseph3d_fwd_tof_sino(
                    xyz1[idx_partial].to(pytomography.device),
                    xyz2[idx_partial].to(pytomography.device),
                    object.to(pytomography.device),
                    self.object_origin,
                    self.object_meta.dr,
                    self.proj_meta.tof_meta.bin_width,
                    self.proj_meta.tof_meta.sigma,
                    self.proj_meta.tof_meta.center_offset,
                    self.proj_meta.tof_meta.n_sigmas,
                    self.proj_meta.tof_meta.num_bins
                ).to(self.output_device)
            else:
                proj[idx_partial] += parallelproj.joseph3d_fwd(
                    xyz1[idx_partial].to(pytomography.device),
                    xyz2[idx_partial].to(pytomography.device),
                    object.to(pytomography.device),
                    self.object_origin,
                    self.object_meta.dr
                ).to(self.output_device)
        N_angles = self.proj_meta.N_angles if subset_idx is None else self.subset_indices_array[subset_idx].shape[0]
        proj = proj.reshape((N_angles, *self.proj_meta.shape[1:], -1))
        if not self.TOF:
            proj=proj.squeeze()
        if self.scale_projection_by_sensitivity:
            proj = proj * (self._compute_sensitivity_sinogram(subset_idx))
            test=self._compute_sensitivity_sinogram(subset_idx)
            # print('system_matrix.forward:self._compute_sensitivity_sinogram(subset_idx)',test.cpu().numpy().max(),test.cpu().numpy().mean(),test.cpu().numpy().min())
        proj = proj.squeeze()  # will remove first dim if nonTOF
        return proj


    def forward2(
            self,
            object: torch.tensor,
            subset_idx: int = None,
    ) -> torch.tensor:
        r"""PET Sinogram forward projection

        Args:
            object (torch.tensor): Object to be forward projected
            subset_idx (int, optional): Subset index for ths sinogram. If None, considers all elements. Defaults to None.
            scale_by_sensitivity (bool, optional): Whether or not to scale the projections by :math:`\mu \eta`. This is not necessarily needed in reconstruction algorithms. Defaults to False.

        Returns:
            torch.tensor: Forward projection
        """
        # Apply object space transforms
        object = object.to(pytomography.device)
        for transform in self.obj2obj_transforms:
            object = transform.forward(object)
        # Project
        xyz1, xyz2 = self._get_xyz_sinogram_coordinates(subset_idx=subset_idx)
        if self.TOF:
            proj = torch.zeros((xyz1.shape[0], self.proj_meta.tof_meta.num_bins)).to(pytomography.dtype).to(
                self.output_device)
        else:
            proj = torch.zeros((xyz1.shape[0])).to(pytomography.dtype).to(self.output_device)
        for idx_partial in torch.tensor_split(torch.arange(xyz1.shape[0]), self.N_splits):
            if self.TOF:
                proj[idx_partial] += parallelproj.joseph3d_fwd_tof_sino(
                    xyz1[idx_partial].to(pytomography.device),
                    xyz2[idx_partial].to(pytomography.device),
                    object.to(pytomography.device),
                    self.object_origin,
                    self.object_meta.dr,
                    self.proj_meta.tof_meta.bin_width,
                    self.proj_meta.tof_meta.sigma,
                    self.proj_meta.tof_meta.center_offset,
                    self.proj_meta.tof_meta.n_sigmas,
                    self.proj_meta.tof_meta.num_bins
                ).to(self.output_device)
            else:
                proj[idx_partial] += parallelproj.joseph3d_fwd(
                    xyz1[idx_partial].to(pytomography.device),
                    xyz2[idx_partial].to(pytomography.device),
                    object.to(pytomography.device),
                    self.object_origin,
                    self.object_meta.dr
                ).to(self.output_device)
        N_angles = self.proj_meta.N_angles if subset_idx is None else self.subset_indices_array[subset_idx].shape[0]
        proj = proj.reshape((N_angles, *self.proj_meta.shape[1:], -1))
        proj = proj.squeeze()  # will remove first dim if nonTOF
        return proj

    def backward(
            self,
            proj: torch.tensor,
            subset_idx: int = None,
            force_scale_by_sensitivity=False,
            force_nonTOF=False,
    ) -> torch.tensor:
        """PET Sinogram back projection

        Args:
            proj (torch.tensor): Sinogram to be back projected
            subset_idx (int, optional): Subset index for ths sinogram. If None, considers all elements. Defaults to None.
            scale_by_sensitivity (bool, optional): Whether or not to scale the projections by :math:`\mu \eta`. This is not necessarily needed in reconstruction algorithms. Defaults to False.
            force_nonTOF (bool, optional): Force non-TOF projection, even if TOF metadata is contained in the projection metadata. This is used for computing normalization factors (which don't depend on TOF). Defaults to False.

        Returns:
            torch.tensor: Back projection.
        """
        # sensitivity scaling

        if force_scale_by_sensitivity or self.scale_projection_by_sensitivity:
            proj = proj * self._compute_sensitivity_sinogram(subset_idx)
        # Project
        xyz1, xyz2 = self._get_xyz_sinogram_coordinates(subset_idx=subset_idx)
        BP = 0
        for idx_partial in torch.tensor_split(torch.arange(xyz1.shape[0]), self.N_splits):
            if self.TOF * (not force_nonTOF):
                BP += parallelproj.joseph3d_back_tof_sino(
                    xyz1[idx_partial].to(pytomography.device),
                    xyz2[idx_partial].to(pytomography.device),
                    self.object_meta.shape,
                    self.object_origin,
                    self.object_meta.dr,
                    proj.flatten(end_dim=-2)[idx_partial].to(pytomography.device),  # flattens to planes,r,theta
                    self.proj_meta.tof_meta.bin_width,
                    self.proj_meta.tof_meta.sigma,
                    self.proj_meta.tof_meta.center_offset,
                    self.proj_meta.tof_meta.n_sigmas,
                    self.proj_meta.tof_meta.num_bins
                )
            else:
                BP += parallelproj.joseph3d_back(
                    xyz1[idx_partial].to(pytomography.device),
                    xyz2[idx_partial].to(pytomography.device),
                    self.object_meta.shape,
                    self.object_origin,
                    self.object_meta.dr,
                    proj.flatten()[idx_partial].to(pytomography.device),  # flattens to planes,r,theta
                )
        # Apply object transforms
        for transform in self.obj2obj_transforms[::-1]:
            BP = transform.backward(BP)
        return BP


def create_sinogramSM_from_LMSM(lm_system_matrix: SystemMatrix, device='cpu'):
    """Generates a sinogram system matrix from a listmode system matrix. This is used in the single scatter simulation algorithm.

    Args:
        lm_system_matrix (SystemMatrix): A listmode PET system matrix
        device (str, optional): The device for any objects in projection space projection space (what it outputs in forward projection and what it expects for back projection). This is seperate from ``pytomography.device`` since the internal functionality may still use GPU even if this is CPU. This is used to save GPU memory since sinograms are often very large. Defaults to pytomography.device.

    Returns:
        SystemMatrix: PET sinogram system matrix generated via a corresponding PET listmode system matrix.
    """
    lm_proj_meta = lm_system_matrix.proj_meta
    sino_proj_meta = PETSinogramPolygonProjMeta(
        lm_proj_meta.info,
        lm_proj_meta.tof_meta
    )
    if lm_proj_meta.weights_sensitivity is not None:
        idxs = torch.arange(lm_proj_meta.scanner_lut.shape[0]).cpu()
        detector_ids_sensitivity = torch.combinations(idxs, 2)
        sinogram_sensitivity = listmode_to_sinogram(
            detector_ids_sensitivity,
            lm_proj_meta.info,
            lm_proj_meta.weights_sensitivity.cpu()
        )
    else:
        sinogram_sensitivity = None
    sino_system_matrix = PETSinogramSystemMatrix(
        lm_system_matrix.object_meta,
        sino_proj_meta,
        obj2obj_transforms=lm_system_matrix.obj2obj_transforms,
        N_splits=20,
        attenuation_map=lm_system_matrix.attenuation_map,
        sinogram_sensitivity=sinogram_sensitivity,
        device=device
    )
    return sino_system_matrix
class MyLikelihood:
    """Generic likelihood class in PyTomography. Subclasses may implement specific likelihoods with methods to compute the likelihood itself as well as particular gradients of the likelihood

    Args:
        system_matrix (SystemMatrix): The system matrix modeling the particular system whereby the projections were obtained
        projections (torch.Tensor | None): Acquired data. If listmode, then this argument need not be provided, and it is set to a tensor of ones. Defaults to None.
        additive_term (torch.Tensor, optional): Additional term added after forward projection by the system matrix. This term might include things like scatter and randoms. Defaults to None.
        additive_term_variance_estimate (Callable, optional): Operator for variance estimate of additive term. If none, then uncertainty estimation does not include contribution from the additive term. Defaults to None.
    """

    def __init__(
            self,
            system_matrix: TOF_PETSystemMatrix,
            projections: torch.Tensor | None = None,
            additive_term: torch.Tensor = None,
            additive_term_variance_estimate: Callable | None = None
    ) -> None:
        self.system_matrix = system_matrix

        if projections is None:  # listmode reconstruction
            self.projections = torch.tensor([1.]).to(pytomography.device)
        else:
            self.projections = projections
        self.FP = None  # stores current state of forward projection
        if type(additive_term) is torch.Tensor:
            self.additive_term = additive_term.to(self.projections.device).to(pytomography.dtype)
            self.exists_additive_term = True
        else:
            self.additive_term = torch.zeros(self.projections.shape).to(self.projections.device).to(pytomography.dtype)
            self.exists_additive_term = False
        self.n_subsets_previous = -1
        self.additive_term_variance_estimate = additive_term_variance_estimate
        self.y_i=torch.zeros(224,449,256).to(self.projections.device).to(pytomography.dtype)
        self.p_i=torch.zeros(224,449,256).to(self.projections.device).to(pytomography.dtype)

    def _set_n_subsets(
            self,
            n_subsets: int
    ) -> None:
        """Sets the number of subsets to be used when computing the likelihood

        Args:
            n_subsets (int): Number of subsets
        """
        self.n_subsets = n_subsets

        if n_subsets < 2:
            self.norm_BP = self.system_matrix.compute_normalization_factor()
        else:
            self.system_matrix.set_n_subsets(n_subsets)
            if self.n_subsets_previous != self.n_subsets:
                self.norm_BPs = []
                for k in range(self.n_subsets):
                    self.norm_BPs.append(self.system_matrix.compute_normalization_factor(k))
        self.n_subsets_previous = n_subsets
    def _update_system_matrix_attenuation(self,new_attenuation_factor):

        #update sinogram_sensitivity
        self.system_matrix.sinogram_sensitivity=new_attenuation_factor.to(self.system_matrix.output_device)
        #update norm_BPs
        self.system_matrix.set_n_subsets(self.n_subsets)
        if self.n_subsets < 2:
            self.norm_BP = self.system_matrix.compute_normalization_factor()
        else:
            self.norm_BPs = []
            for k in range(self.n_subsets):
                self.norm_BPs.append(self.system_matrix.compute_normalization_factor(k))


    def _get_projection_subset(self, projections: torch.Tensor, subset_idx: int | None = None) -> torch.Tensor:
        """Method for getting projection subset corresponding to given subset index

        Args:
            projections (torch.Tensor): Projection data
            subset_idx (int): Subset index

        Returns:
            torch.Tensor: Subset projection data
        """
        if subset_idx is None:
            return projections
        else:
            return self.system_matrix.get_projection_subset(projections, subset_idx)

    def _get_normBP(self, subset_idx: int, return_sum: bool = False):
        """Gets normalization factor (back projection of ones)

        Args:
            subset_idx (int): Subset index
            return_sum (bool, optional): Sum normalization factor from all subsets. Defaults to False.

        Returns:
            torch.Tensor: Normalization factor
        """
        if subset_idx is None:
            return self.norm_BP
        else:
            if return_sum:
                return torch.stack(self.norm_BPs).sum(axis=0)
            else:
                # Put on PyTomography device in case stored on CPU
                return self.norm_BPs[subset_idx].to(pytomography.device)


    def compute_gradient(
            self,
            object: torch.Tensor,
            subset_idx: int | None = None,
            norm_BP_subset_method: str = 'subset_specific'
    ) -> torch.Tensor:
        r"""Computes the gradient for the Poisson log likelihood given by :math:`\nabla_f L(g|f) =  H^T (g / Hf) - H^T 1`.

        Args:
            object (torch.Tensor): Object :math:`f` on which the likelihood is computed
            subset_idx (int | None, optional): Specifies the subset for forward/back projection. If none, then forward/back projection is done over all subsets, and the entire projections :math:`g` are used. Defaults to None.
            norm_BP_subset_method (str, optional): Specifies how :math:`H^T 1` is calculated when subsets are used. If 'subset_specific', then uses :math:`H_m^T 1`. If `average_of_subsets`, then uses the average of all :math:`H_m^T 1`s for any given subset (scaled to the relative size of the subset if subsets are not equal size). Defaults to 'subset_specific'.

        Returns:
            torch.Tensor: The gradient of the Poisson likelihood.
        """
        proj_subset = self._get_projection_subset(self.projections, subset_idx) #y_it
        additive_term_subset = self._get_projection_subset(self.additive_term, subset_idx) #s_it
        self.projections_predicted = self.system_matrix.forward(object, subset_idx) + additive_term_subset #a*p_it
        norm_BP = self._get_normBP(subset_idx)
        return self.system_matrix.backward(proj_subset / (self.projections_predicted + pytomography.delta),
                                           subset_idx) - norm_BP

    def compute_gradient_ff(
            self,
            object: torch.Tensor,
            precomputed_forward_projection: torch.Tensor | None = None,
            subset_idx: int = None,
    ) -> Callable:
        r"""Computes the second order derivative :math:`\nabla_{ff} L(g|f) = -H^T (g/(Hf+s)^2) H`.

        Args:
            object (torch.Tensor): Object :math:`f` used in computation.
            precomputed_forward_projection (torch.Tensor | None, optional): The quantity :math:`Hf`. If this value is None, then the forward projection is recomputed. Defaults to None.
            subset_idx (int, optional): Specifies the subset for all computations. Defaults to None.

        Returns:
            Callable: The operator given by the second order derivative.
        """
        if precomputed_forward_projection is None:
            FP = self.system_matrix.forward(object, subset_idx)
        else:
            FP = precomputed_forward_projection
        proj_subset = self._get_projection_subset(self.projections, subset_idx)

        def operator(input):
            input = self.system_matrix.forward(input, subset_idx)
            input = input * proj_subset / (FP ** 2 + pytomography.delta)
            return -self.system_matrix.backward(input, subset_idx)

        return operator

    def compute_gradient_gf(
            self,
            object,
            precomputed_forward_projection=None,
            subset_idx=None,
    ):
        r"""Computes the second order derivative :math:`\nabla_{gf} L(g|f) = 1/(Hf+s) H`.

        Args:
            object (torch.Tensor): Object :math:`f` used in computation.
            precomputed_forward_projection (torch.Tensor | None, optional): The quantity :math:`Hf`. If this value is None, then the forward projection is recomputed. Defaults to None.
            subset_idx (int, optional): Specifies the subset for all computations. Defaults to None.

        Returns:
            Callable: The operator given by the second order derivative.
        """
        if precomputed_forward_projection is None:
            FP = self.system_matrix.forward(object, subset_idx)
        else:
            FP = precomputed_forward_projection

        def operator(input):
            input = self.system_matrix.forward(input, subset_idx)
            return input / (FP + pytomography.delta)

        return operator

    def compute_gradient_sf(
            self,
            object,
            precomputed_forward_projection=None,
            subset_idx=None,
    ):
        r"""Computes the second order derivative :math:`\nabla_{sf} L(g|f,s) = -g/(Hf+s)^2 H` where :math:`s` is an additive term representative of scatter.

        Args:
            object (torch.Tensor): Object :math:`f` used in computation.
            precomputed_forward_projection (torch.Tensor | None, optional): The quantity :math:`Hf`. If this value is None, then the forward projection is recomputed. Defaults to None.
            subset_idx (int, optional): Specifies the subset for all computations. Defaults to None.

        Returns:
            Callable: The operator given by the second order derivative.
        """
        proj_subset = self._get_projection_subset(self.projections, subset_idx)
        if precomputed_forward_projection is None:
            FP = self.system_matrix.forward(object, subset_idx)
        else:
            FP = precomputed_forward_projection

        def operator(input):
            input = self.system_matrix.forward(input, subset_idx)
            return -input * proj_subset / (FP + pytomography.delta) ** 2

        return operator