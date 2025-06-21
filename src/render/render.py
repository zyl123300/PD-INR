import torch
import torch.nn as nn
from pandas.tests.indexes.test_base import test_validate_1d_input
from sympy.physics.units import action


def render_with_box(rays, net, net_fine, n_samples, n_fine, perturb, netchunk, raw_noise_std):
    near = rays[..., 6]
    far = rays[..., 7]

    mask_non_empty = near < far

    acc = torch.zeros_like(near)
    if torch.any(mask_non_empty):
        acc_non_empty = \
        render(rays[mask_non_empty], net, net_fine, n_samples, n_fine, perturb, netchunk, raw_noise_std)["acc"]
        # acc_non_empty = render2(rays[mask_non_empty], net, netchunk, divide = 128)["acc"]
        acc[mask_non_empty] = acc_non_empty
    ret = {"acc": acc}
    return ret


def render2(rays, net, netchunk, divide=192):
    # TO DO:
    # making parallel
    # perturb

    delta_t = net.bound / divide
    n_rays = rays.shape[0]
    rays_o, rays_d, near, far = rays[..., :3], rays[..., 3:6], rays[..., 6], rays[..., 7]

    pts_list = []
    dists_list = []
    split_len_list = []
    acc = torch.zeros_like(near)
    for i in range(n_rays):
        # t_vals = torch.arange(near[i], far[i], delta_t, device = near.device)
        t_vals = torch.linspace(0., 1., steps=max(int((far[i] - near[i]) / delta_t) + 1, 2), device=near.device)
        z_vals = near[i] * (1. - t_vals) + far[i] * t_vals
        mids = .5 * (z_vals[1:] + z_vals[:-1])

        pts = rays_o[i] + rays_d[None, i] * mids[:, None]
        pts_list.append(pts)

        dists = torch.norm(rays_d[i]) * (z_vals[1:] - z_vals[:-1])
        # acc[i] = torch.sum(dists * net(pts))
        dists_list.append(dists)
        split_len_list.append(pts.shape[0])

    uvt_flat = torch.cat(pts_list, 0)
    raw = torch.cat([net(uvt_flat[i:i + netchunk]) for i in range(0, uvt_flat.shape[0], netchunk)], 0).squeeze()
    raw_split = torch.split(raw, split_len_list)
    # print(raw.shape)
    for i in range(n_rays):
        # print(dists_list[i].shape)
        acc[i] = torch.sum(raw_split[i] * dists_list[i])

    return {"acc": acc}


def render_tof(rays, tof_meta, net, net_fine, n_samples, n_fine, perturb, netchunk, raw_noise_std):
    n_rays = rays.shape[0]  # n_rays
    rays_o, rays_d, near, far = rays[..., :3], rays[..., 3:6], rays[..., 6:7], rays[..., 7:]
    middle = torch.ones_like(near) * 0.5
    # TOF 相关参数
    bin_width = tof_meta.bin_width.to(rays.device) / 1000  # 23.07mm

    sigma = tof_meta.sigma.to(rays.device) / 1000  # 12.73mm

    bin_positions = tof_meta.bin_positions.to(rays.device) / 1000
    n_sigmas = tof_meta.n_sigmas  # 3
    num_bins = tof_meta.num_bins
    n_tof = num_bins

    sigma_eff = torch.sqrt(sigma ** 2 + bin_width ** 2 / 12.0)
    tof_trunc_corr_factor = 1.0 / torch.erf(n_sigmas / torch.sqrt(torch.tensor(2.0)))
    # 改变参考系,不同射线下长度度量不同
    t_sigma = sigma / torch.norm(rays_d, dim=-1, keepdim=True)
    t_bin_positions = bin_positions.expand(n_rays, num_bins) / torch.norm(rays_d, dim=-1,
                                                                          keepdim=True) + middle  # (n_rays,n_tof)
    t_sigma_eff = sigma_eff / torch.norm(rays_d, dim=-1, keepdim=True)
    t_bin_width = bin_width / torch.norm(rays_d, dim=-1, keepdim=True)
    # 计算采样点的位置
    t_vals = torch.linspace(0., 1., steps=n_samples, device=near.device).unsqueeze(0)
    z_vals = near * (1. - t_vals) + far * t_vals  # (n_rays, n_samples)
    z_vals = z_vals.expand([n_rays, n_samples])  # (n_rays, n_samples)

    if perturb:
        # 分段采样，增加随机扰动
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])  # (n_rays, n_samples-1)
        upper = torch.cat([mids, z_vals[..., -1:]], -1)  # (n_rays, n_samples)
        lower = torch.cat([z_vals[..., :1], mids], -1)  # (n_rays, n_samples)
        t_rand = torch.rand(z_vals.shape, device=lower.device)
        z_vals = lower + (upper - lower) * t_rand  # (n_rays, n_samples)

    # 计算每个采样点的位置
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # (n_rays, n_samples, 3)
    # raw = run_network(pts, net, netchunk)  # (n_rays, n_samples, 1)
    raw, tv_gradient2 = my_run_network(pts, net, netchunk, z_vals, rays_d, )

    dtof = torch.abs(z_vals[..., None] - t_bin_positions[..., None, :])  # (n_rays, n_samples, n_tof)

    t_bin_width = t_bin_width.unsqueeze(-1).expand(dtof.shape)  # (n_rays, n_samples, n_tof)
    t_sigma = t_sigma.unsqueeze(-1).expand(dtof.shape)  # (n_rays, n_samples, n_tof)
    t_sigma_eff = t_sigma_eff.unsqueeze(-1).expand(dtof.shape)  # (n_rays, n_samples, n_tof)
    tof_weights = 0.5 * (
            torch.erf((dtof + 0.5 * t_bin_width) / (torch.sqrt(torch.tensor(2.0)) * t_sigma)) -
            torch.erf((dtof - 0.5 * t_bin_width) / (torch.sqrt(torch.tensor(2.0)) * t_sigma)))
    ###不均匀体素的上下边界
    upper_bound = torch.ones((n_rays, n_samples)).to(rays.device)
    upper_bound[..., -1:] = far
    upper_bound[..., 0:-1] = (z_vals[..., :-1] + z_vals[..., 1:]) / 2.0

    lower_bound = torch.zeros((n_rays, n_samples)).to(rays.device)
    lower_bound[..., :1] = near
    lower_bound[..., 1:] = (z_vals[..., :-1] + z_vals[..., 1:]) / 2.0

    upper_bound_distance_tof = torch.abs(upper_bound[..., None] - t_bin_positions[..., None, :])
    lower_bound_distance_tof = torch.abs(lower_bound[..., None] - t_bin_positions[..., None, :])
    ########
    tof_valid_mask = (upper_bound_distance_tof < n_sigmas * t_sigma_eff) | (
                lower_bound_distance_tof < n_sigmas * t_sigma_eff)
    tof_weights = tof_weights * tof_valid_mask  # 截断
    tof_weights *= tof_trunc_corr_factor
    # 对 TOF 权重进行归一化，使每个采样点在所有 TOF bin 上的权重和为 1
    # tof_weights_sum = tof_weights.sum(dim=-1, keepdim=True)  # (n_rays, n_samples, 1)
    # tof_weights = tof_weights / (tof_weights_sum + 1e-8)  # 避免除以 0
    # 计算经过 TOF 加权后的累积权重和颜色
    acc, tv_gradient = raw2outputs_tof(raw, z_vals, rays_d, tof_weights, raw_noise_std)

    ret = {"acc": acc, "pts": pts, "tv_gradient": tv_gradient, "tv_gradient2": tv_gradient2, }
    return ret


def raw2outputs_tof(raw, z_vals, rays_d, tof_weights, raw_noise_std=0.):
    # 计算相邻采样点之间的距离 (梯形积分)
    dists = z_vals * torch.norm(rays_d[..., None, :], dim=-1)  # (n_rays, n_samples)
    raw_ = raw.squeeze()

    # 梯形积分计算累积的 acc_map，加入 TOF 权重
    acc = raw_[..., 0].unsqueeze(-1) * (dists[..., 1] - dists[..., 0]).unsqueeze(-1) * tof_weights[..., 0, :] + raw_[
        ..., -1].unsqueeze(-1) * (
                  dists[..., -1] - dists[..., -2]).unsqueeze(-1) * tof_weights[..., -1, :]
    acc += torch.sum(
        raw_[..., 1:-1].unsqueeze(-1) * (dists[..., 2:] - dists[..., :-2]).unsqueeze(-1) * tof_weights[..., 1:-1, :],
        dim=-2) / 2  # (n_rays,n_tof)
    # TV
    tv_gradient = torch.mean((torch.abs(raw_[..., 1:] - raw_[..., : -1])) * (dists[..., 1:] - dists[..., : -1]),
                             dim=-1)  # (n_rays,)

    return acc, tv_gradient


def render(rays, net, net_fine, n_samples, n_fine, perturb, netchunk, raw_noise_std):
    n_rays = rays.shape[0]
    rays_o, rays_d, near, far = rays[..., :3], rays[..., 3:6], rays[..., 6:7], rays[..., [7]]

    t_vals = torch.linspace(0., 1., steps=n_samples, device=near.device)
    z_vals = near * (1. - t_vals) + far * (t_vals)
    # print("t", near.shape)
    # print("z", z_vals.shape)
    z_vals = z_vals.expand([n_rays, n_samples])  # (1024,256)

    if perturb:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])  # (1024,255)
        upper = torch.cat([mids, z_vals[..., -1:]], -1)  # (1024,256)
        lower = torch.cat([z_vals[..., :1], mids], -1)  # (1024,256)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=lower.device)
        z_vals = lower + (upper - lower) * t_rand  # (1024,256)

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [n_rays, n_samples, 3]

    if rays.shape[-1] > 8:
        # print("use phase")
        phase = rays[..., 8]
        pts = torch.cat([pts, phase.expand(n_samples, n_rays).T.unsqueeze(-1)], dim=-1)

    bound = net.bound - 1e-8
    pts = pts.clamp(-bound, bound)

    raw = run_network(pts, net, netchunk)
    acc, weights = raw2outputs(raw, z_vals, rays_d, raw_noise_std)

    if net_fine is not None and n_fine > 0:
        acc_0 = acc
        weights_0 = weights
        pts_0 = pts

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_fine, det=(perturb == 0.))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts = pts.clamp(-bound, bound)
        raw = run_network(pts, net_fine, netchunk)
        acc, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std)

    ret = {"acc": acc, "pts": pts}
    if net_fine is not None and n_fine > 0:
        ret["acc0"] = acc_0
        ret["weights0"] = weights_0
        ret["pts0"] = pts_0

    for k in ret:
        if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def run_network(inputs, fn, netchunk):
    """
    Prepares inputs and applies network "fn".
    """
    # print(inputs.shape)
    uvt_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # (n_rays * n_samples, 3)
    out_flat = torch.cat([fn(uvt_flat[i:i + netchunk]) for i in range(0, uvt_flat.shape[0], netchunk)],
                         0)  # (n_rays * n_samples, n_channels)
    out = out_flat.reshape(list(inputs.shape[:-1]) + [out_flat.shape[-1]])

    return out


def my_run_network(inputs, fn, netchunk, z_vals, rays_d, ):
    dists = z_vals * torch.norm(rays_d[..., None, :], dim=-1)
    uvt_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # (n_rays * n_samples, 3)
    feature_flat = []
    for i in range(0, uvt_flat.shape[0], netchunk):
        feature_i = fn.encoder(uvt_flat[i:i + netchunk], fn.bound)  # (n_rays,n_samples,n_feature)
        feature_flat.append(feature_i)
    feature_flat = torch.cat(feature_flat, dim=0)
    feature = feature_flat.reshape(list(inputs.shape[:-1]) + [feature_flat.shape[-1]])  # (n_rays,n_samples,n_features)
    tv_gradient2 = torch.mean(
        torch.mean(torch.abs(feature[:, 1:, :8] - feature[:, :-1, :8]), dim=-1) * (dists[..., 1:] - dists[..., : -1]),
        dim=-1)
    out_flat = torch.cat([fn.density(feature_flat[i:i + netchunk]) for i in range(0, feature_flat.shape[0], netchunk)],
                         dim=0)
    out = out_flat.reshape(list(inputs.shape[:-1]) + [out_flat.shape[-1]])

    return out, tv_gradient2


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.):
    """Transforms model"s predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # trapezoidal integration
    dists = z_vals * torch.norm(rays_d[..., None, :], dim=-1)  # (1024,256,1)
    raw_ = raw.squeeze()
    acc = raw_[..., 0] * (dists[..., 1] - dists[..., 0]) + raw_[..., -1] * (dists[..., -1] - dists[..., -2])
    acc += torch.sum(raw_[..., 1:-1] * (dists[..., 2:] - dists[..., :-2]), dim=-1) / 2

    if raw.shape[-1] == 1:
        eps = torch.ones_like(raw[:, :1, -1]) * 1e-10
        weights = torch.cat([eps, torch.abs(raw[:, 1:, -1] - raw[:, :-1, -1])], dim=-1)
        weights = weights / torch.max(weights)
    elif raw.shape[-1] == 2:  # with jac
        weights = raw[..., 1] / torch.max(raw[..., 1])
    else:
        raise NotImplementedError("Wrong raw shape")

    return acc, weights


def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous().to(cdf.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
