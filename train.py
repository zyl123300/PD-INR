import os.path as osp
import torch
import numpy as np
import argparse
from src.config.configloading import load_config
from src.trainer import TOFTrainer

work_dictionary = 'experiments/'
expdir = osp.join(work_dictionary, 'tof_pet/logs/')
datadir = 'your_dataset.pickle' #replace with your dataset

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/basic.yaml",
                        help="configs file path")
    parser.add_argument('--gpu_index', default=0, type=int, help='gpu index.')
    parser.add_argument('--expname', default="PD-INR", type=str)
    parser.add_argument('--expdir', default=expdir, type=str)
    parser.add_argument('--datadir', default=datadir, type=str)
    return parser

parser = config_parser()
args = parser.parse_args()

cfg = load_config(args.config)
cfg["exp"] = dict()
cfg["exp"]["expname"] = args.expname
cfg["exp"]["expdir"] = args.expdir
cfg["exp"]["datadir"] = args.datadir
cfg["encoder"]["prior_image_path"] = 'your_prior_img.bin' #path to your prior image
cfg["exp"]["tv_weight"] = 5e-5

# Run on GPU if CUDA is available
RUN_ON_GPU = torch.cuda.is_available()

# # Set random seeds
SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)

if RUN_ON_GPU:
    torch.cuda.empty_cache()
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

if RUN_ON_GPU:
    device = torch.device(index=args.gpu_index, type='cuda')
else:
    device = torch.device('cpu')

trainer = TOFTrainer(cfg, device)

trainer.start()
