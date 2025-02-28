
def get_encoder(voxels, sVoxel,prior_image_path, voxels_shape, encoding, input_dim=3,
                multires=6,
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19,
                base_resolution_time=4, base_reso_max=256,
                **kwargs):
    if encoding == "None":
        return lambda x, **kwargs: x, input_dim

    elif encoding == "frequency":
        from .freqencoder import FreqEncoder
        encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires - 1, N_freqs=multires, log_sampling=True)

    elif encoding == "hashgrid":
        from .hashencoder import HashGridEncoder
        encoder = HashGridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim,
                                  base_resolution=base_resolution,
                                  log2_hashmap_size=log2_hashmap_size)
    elif encoding == 'priorhashgrid':
        from .hashencoder import ImagePriorHashEncoder
        encoder = ImagePriorHashEncoder(voxels=voxels, sVoxel=sVoxel, prior_image_path=prior_image_path, voxels_shape= voxels_shape,input_dim=input_dim, num_levels=num_levels,
                                        level_dim=level_dim,
                                        base_resolution=base_resolution,
                                        log2_hashmap_size=log2_hashmap_size)



    else:
        raise NotImplementedError()
    return encoder
