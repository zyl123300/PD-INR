
import json
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from shutil import copyfile

from . import get_mse_log, render_ct, calc_TV_loss, calc_TV2_loss
from .dataset.add_poisson_noise import bi_load
from .dataset.ct import CTDataset
# import dataset.pet
from src.dataset.tofpet import TOFPETDataset
# from src.dataset.pet import PETDataset
from .network import get_network
from .encoder import get_encoder
import os
import os.path as osp
import torch
# import imageio.v2 as iio
import cv2
import numpy as np
import argparse

# from src.config.configloading import load_config
from .render import render, render_tof, run_network, render_utof, render_atof, render_with_box, render2
from .loss import calc_mse_loss, calc_poisson_loss, calc_tv_loss, calc_huber_loss, calc_gmc_loss
from .utils import get_psnr, get_mse, get_psnr_3d, get_ssim_3d, cast_to_image


class Trainer:
    def __init__(self, cfg, device="cuda"):
        # Args
        self.global_step = 0
        self.conf = cfg
        self.n_fine = cfg["render"]["n_fine"]
        self.epochs = cfg["train"]["epoch"]
        self.i_eval = cfg["log"]["i_eval"]
        self.i_save = cfg["log"]["i_save"]
        self.netchunk = cfg["render"]["netchunk"]
        self.n_rays = cfg["train"]["n_rays"]

        if "use_OS" in cfg["train"].keys():
            self.use_OS = cfg["train"]["use_OS"]
        else:
            self.use_OS = False

        if self.use_OS:
            print("Use ordered subset method")
        else:
            print("Use random sampling")

        self.batch_size = cfg["train"]["n_batch"]

        # Log direcotry
        self.expdir = osp.join(cfg["exp"]["expdir"], cfg["exp"]["expname"])
        self.ckptdir = osp.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.evaldir = osp.join(self.expdir, "eval")
        os.makedirs(self.evaldir, exist_ok=True)

        self.train_dset = TOFPETDataset(cfg["exp"]["datadir"], n_rays=cfg["train"]["n_rays"], device=device)
        self.tof_meta = self.train_dset.tof_meta
        self.voxels = self.train_dset.voxels  # if self.i_eval > 0 else None
        self.voxels_shape =self.train_dset.voxels_shape
        # Network
        network = get_network(cfg["network"]["net_type"])
        cfg["network"].pop("net_type", None)
        encoder = get_encoder(voxels=self.voxels, sVoxel=self.train_dset.sVoxel, voxels_shape=self.voxels_shape, **cfg["encoder"]).to(device)
        self.net = network(encoder, sVoxel=self.train_dset.sVoxel, n_samples=cfg["render"]["n_samples"],
                           **cfg["network"]).to(device)

        grad_vars = list(self.net.parameters())

        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print("Total number of parameters: ", total_params)

        self.net_fine = None
        if self.n_fine > 0:
            self.net_fine = network(encoder, **cfg["network"]).to(device)
            grad_vars += list(self.net_fine.parameters())

        # Optimizer
        # self.optimizer = torch.optim.Adam(params=grad_vars, betas=(0.9, 0.999))
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer=self.optimizer, step_size=cfg["train"]["lrate_step"], gamma=cfg["train"]["lrate_gamma"])

        self.encoding_method = cfg["encoder"]["encoding"]
        if "grid_planes" == self.encoding_method:
            print("AdamW optimizer")
            self.optimizer = torch.optim.AdamW(params=[{'params': self.net.encoder.planes, 'weight_decay': 0},
                                                       {'params': self.net.encoder.encoder.parameters(),
                                                        'weight_decay': 1e-4},
                                                       {'params': self.net.density.parameters(), 'weight_decay': 1e-6}],
                                               lr=cfg["train"]["lrate"], amsgrad=True)

            # params_list_weight = (p for name, p in self.net.density.named_parameters() if 'bias' not in name)
            # params_list_bias = (p for name, p in self.net.density.named_parameters() if 'bias' in name)
            # self.optimizer = torch.optim.AdamW(params = [{'params': self.net.encoder.planes, 'weight_decay': 0},
            #                                     {'params': self.net.encoder.encoder.parameters(), 'weight_decay': 1e-2},
            #                                     {'params': params_list_weight, 'weight_decay': 1e-3},
            #                                     {'params': params_list_bias}],
            #                                     lr = cfg["train"]["lrate"], amsgrad = True)
        elif "hashgrid" == self.encoding_method:
            print("AdamW optimizer")
            # self.optimizer = torch.optim.AdamW(params = [{'params': self.net.encoder.encoder.parameters(), 'weight_decay': 1e-4},
            #                                     {'params': self.net.density.parameters(), 'weight_decay': 1e-6}],
            #                                     lr = cfg["train"]["lrate"], amsgrad = True)
            self.optimizer = torch.optim.Adam(params=grad_vars, lr=cfg["train"]["lrate"], betas=(0.9, 0.999),
                                              weight_decay=0.0, amsgrad=True)
        else:
            print("AdamW optimizer")
            self.optimizer = torch.optim.Adam(params=grad_vars, lr=cfg["train"]["lrate"], betas=(0.9, 0.999),
                                              weight_decay=0.0, amsgrad=True)

        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
        #                             cfg["train"]["epoch"],
        #                             cfg["train"]["lrate"] / 3)

        if "pretrained" in cfg["train"].keys():
            ckpt = torch.load(cfg["train"]["pretrained"])
            # print(ckpt["network"].keys())
            state_dict = dict()
            for k, v in ckpt["network"].items():
                if "density" in k:
                    state_dict[k] = v
            state_dict_current = self.net.state_dict()
            state_dict_current.update(state_dict)
            self.net.load_state_dict(state_dict_current)

        # Load checkpoints
        self.epoch_start = 0
        if cfg["train"]["resume"] and osp.exists(self.ckptdir):
            print(f"Load checkpoints from {self.ckptdir}.")
            ckpt = torch.load(self.ckptdir)
            self.epoch_start = ckpt["epoch"] + 1
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.global_step = self.epoch_start * len(self.train_dset)
            self.net.load_state_dict(ckpt["network"])
            if self.n_fine > 0:
                self.net_fine.load_state_dict(ckpt["network_fine"])

        # Summary writer
        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text("parameters", self.args2string(cfg), global_step=0)

    def args2string(self, hp):
        """
        Transfer args to string.
        """
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

    def start(self):
        """
        Main loop.
        """

        def fmt_loss_str(losses):
            return "".join(", " + k + ": " + f"{losses[k].item():.3g}" for k in losses)

        iter_per_epoch = len(self.train_dset)
        pbar = tqdm(total=iter_per_epoch * self.epochs, leave=True)
        if self.epoch_start > 0:
            pbar.update(self.epoch_start * iter_per_epoch)

        record = []
        for idx_epoch in range(self.epoch_start + 1, self.epochs + 1):
            timein = time.time()
            self.train_dset.update_indices()
            train_dloader = torch.utils.data.DataLoader(self.train_dset, batch_size=self.batch_size)

            # if "grid" in self.encoding_method:
            # if idx_epoch == 3:
            #     for param in self.net.encoder.parameters():
            #         param.requires_grad = False

            # Train
            loss_train_list = []
            for data in train_dloader:
                self.global_step += 1
                # Train
                self.net.train()
                loss_train = self.train_step(data, global_step=self.global_step, idx_epoch=idx_epoch)
                pbar.set_description(
                    f"epoch={idx_epoch}/{self.epochs}, loss={loss_train:.3g}, lr={self.optimizer.param_groups[0]['lr']:.3g}")
                pbar.update(1)

                loss_train_list.append(loss_train)
            record.append([idx_epoch, np.mean(loss_train_list)])

            # Evaluate
            if (idx_epoch % self.i_eval == 0 or idx_epoch == self.epochs) and (self.i_eval > 0) and (idx_epoch > 0):
                self.net.eval()
                with torch.no_grad():
                    loss_test = self.eval_step(global_step=self.global_step, idx_epoch=idx_epoch)
                self.net.train()
                tqdm.write(f"[EVAL] epoch: {idx_epoch}/{self.epochs}{fmt_loss_str(loss_test)}")

            # Save
            if (idx_epoch % self.i_save == 0 or idx_epoch == self.epochs) and self.i_save > 0 and idx_epoch > 0:
                if osp.exists(self.ckptdir):
                    copyfile(self.ckptdir, self.ckptdir_backup)
                tqdm.write(f"[SAVE] epoch: {idx_epoch}/{self.epochs}, path: {self.ckptdir}")
                torch.save(
                    {
                        "epoch": idx_epoch,
                        "network": self.net.state_dict(),
                        "network_fine": self.net_fine.state_dict() if self.n_fine > 0 else None,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    self.ckptdir,
                )

                np.savetxt(os.path.join(self.expdir, "loss.txt"), np.array(record))

            # Update lrate
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            # self.lr_scheduler.step()
            print(f'time spend:{time.time() - timein}')
        tqdm.write(f"Training complete! See logs in {self.expdir}")

    def train_step(self, data, global_step, idx_epoch):
        """
        Training step
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, global_step, idx_epoch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_loss(self, data, global_step, idx_epoch):
        """
        Training step
        """
        raise NotImplementedError()

    def eval_step(self, global_step, idx_epoch):
        """
        Evaluation step
        """
        raise NotImplementedError()

class TOFTrainer(Trainer):
    def __init__(self,cfg: dict , device):
        """
        Basic network trainer.
        """
        super().__init__(cfg, device)

    def compute_loss(self, data, global_step, idx_epoch):
        projs = data["projs"].reshape(-1, 13)
        rays = data["rays"].reshape(-1, 8)
        ret = render_tof(rays, tof_meta=self.tof_meta, net=self.net,net_fine=self.net_fine, **self.conf["render"])

        projs_pred = ret["acc"]  #(n_rays,n_tof)
        # tv_gradient = ret["tv_gradient"] #(n_rays,)
        tv_gradient2 = ret["tv_gradient2"]
        loss = {"loss": 0.}
        # calc_mse_loss(loss, projs, projs_pred)
        calc_poisson_loss(loss, projs_pred, projs, lam=1)

        # calc_TV_loss(loss,tv_gradient, gamma= 2*1e-5)
        # loss["loss"] = loss["loss_poisson"] + loss["tv_gradient"]

        calc_TV2_loss(loss,tv_gradient2, gamma= 8*1e-6) #平衡的指标=2*1e-5    1*1e-5
        loss["loss"] = loss["loss_poisson"] + loss["tv_gradient2"]
        # Log
        for ls in loss.keys():
            self.writer.add_scalar(f"train/{ls}", loss[ls].item(), global_step)
        return loss["loss"]

    def eval_step(self, global_step, idx_epoch):

        image = self.train_dset.image
        image = image.squeeze()

        # image = (image - image.min()) / (image.max() - image.min())
        voxels = self.train_dset.voxels

        image_pred = run_network(voxels, self.net_fine if self.net_fine is not None else self.net, self.netchunk)
        image_pred = image_pred.squeeze()
        # image_pred = (image_pred - image_pred.min()) / (image_pred.max() - image_pred.min())

        loss = {
            # "proj_mse": get_mse(projs_pred, projs),
            # "proj_psnr": get_psnr(projs_pred, projs),
            "mse_log": get_mse_log(image, image_pred),
            "psnr_3d": get_psnr_3d(image_pred, image),
            "ssim_3d": get_ssim_3d(image_pred, image),
        }

        # Logging
        show_slice = 5

        show_step = image.shape[-1] // show_slice
        show_image = image[..., ::show_step]
        show_image_pred = image_pred[..., ::show_step]
        show = []
        for i_show in range(show_slice):
            show.append(torch.concat([show_image[..., i_show], show_image_pred[..., i_show]], dim=0))
        show_density = torch.concat(show, dim=1)
        self.writer.add_image("eval/density (row1: gt, row2: pred)", cast_to_image(show_density), global_step,
                              dataformats="HWC")

        # show_proj = torch.concat([projs, projs_pred], dim=1)
        # self.writer.add_image("eval/projection (left: gt, right: pred)", cast_to_image(show_proj), global_step, dataformats="HWC")

        for ls in loss.keys():
            self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)

        # Save
        eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        bi_save(image_pred.cpu().detach().numpy(), osp.join(eval_save_dir, f"image_pred_{idx_epoch}.bin"))


        cv2.imwrite(osp.join(eval_save_dir, "slice_show_row1_gt_row2_pred.png"),
                    (cast_to_image(show_density) * 255).astype(np.uint8))

        with open(osp.join(eval_save_dir, "stats.txt"), "w") as f:
            for key, value in loss.items():
                f.write("%s: %f\n" % (key, value.item()))

        return loss
    def update_net_encoder(self, image):
        self.net.encoder.update_image_prior(image)