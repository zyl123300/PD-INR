import numpy as np
import pytomography
import torch
from pytomography.io.PET import gate
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETTOFMeta, PETSinogramPolygonProjMeta

def bi_save(data: np.ndarray, path: str):
    if isinstance(data,torch.Tensor):
        data=data.cpu().numpy()
    bi_file = open(path, 'wb')
    data=data.astype(np.float32)
    bi_file.write(np.flip(np.transpose(data, (2, 1, 0)), axis=[0, 2]).copy(order='C'))
   # bi_file.write(full_patch_slice.copy(order='C'))
    bi_file.close()

def bi_load(path: str, shape: tuple):
    shape = tuple(reversed(shape))
    data = (np.fromfile(path, dtype='float32').reshape(shape).astype(np.float32))
    data=np.flip(np.transpose(data, (2, 1, 0)), axis=[0, 2]).copy()
    return data

def bi_save4(data: np.ndarray, path: str):
    bi_file = open(path, 'wb')
    data = data.astype(np.float16)
    bi_file.write(np.flip(np.transpose(data, (2, 1, 0, 3)), axis=[0, 2]).copy(order='C'))
        # bi_file.write(full_patch_slice.copy(order='C'))
    bi_file.close()

def bi_load4(path: str, shape: tuple):
    shape2 = (shape[2], shape[1], shape[0], shape[3])
    data = (np.fromfile(path, dtype='float16').reshape(shape2).astype(np.float16))
    data=np.flip(np.transpose(data, (2, 1, 0, 3)), axis=[0, 2]).copy()
    return data

def get_psnr_3d(arr1, arr2, size_average=False,
                FOV_mask=None):
    """
    :param arr1:
        Format-[NDHW], OriImage [0,1]
    :param arr2:
        Format-[NDHW], ComparedImage [0,1]
    :return:
        Format-None if size_average else [N]
    """

    if FOV_mask is not None:
        arr1 = arr1[FOV_mask].copy()
        arr2 = arr2[FOV_mask].copy()

    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    PIXEL_MAX = arr2.max()
    # arr1 = arr1[np.newaxis, ...]
    # arr2 = arr2[np.newaxis, ...]
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    # eps = 1e-10
    se = np.power(arr1 - arr2, 2)
    # mse = se.mean(axis=1).mean(axis=1).mean(axis=1)
    mse = se.mean()
    # zero_mse = np.where(mse == 0)
    # mse[zero_mse] = eps
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    # #zero mse, return 100
    # psnr[zero_mse] = 100

    if size_average:
        return psnr.mean()
    else:
        return psnr


def get_ssim_3d(arr1, arr2, size_average=True, data_range=1.0,
                FOV_mask=None):
    """
    :param arr1:
        Format-[NDHW], OriImage [0,1]
    :param arr2:
        Format-[NDHW], ComparedImage [0,1]
    :return:
        Format-None if size_average else [N]
    """
    # if FOV_mask is not None:
    #     arr1 = arr1.copy()
    #     arr2 = arr2.copy()
    #     arr1[~FOV_mask] = 0
    #     arr2[~FOV_mask] = 0

    # truncate to [0, 1]
    # arr1 = np.maximum(np.minimum(arr1, 1), 0)
    # arr2 = np.maximum(np.minimum(arr2, 1), 0)
    # arr1 = np.clip(arr1, 0, 1)
    # arr2 = np.clip(arr2, 0, 1)

    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)

    N = arr1.shape[0]

    _, ssim = structural_similarity(arr1[0], arr2[0], full=True,
                                    data_range=data_range)

    if FOV_mask is not None:
        ssim = ssim[FOV_mask]

    return ssim.mean()




