from __future__ import print_function, division

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.tri_raft_stereo import TriRAFTStereo

from evaluate_stereo import *
import core.stereo_datasets as datasets

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5" 

import psutil

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler, logdir):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.logdir=logdir
        self.writer = SummaryWriter(log_dir=self.logdir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.logdir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.logdir)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def depth2disparity(depth, intri, extri, other_extri = None):
    # depth: [B, H, W]
    # intri: [B, 3, 3]
    # extri: [B, 4, 4]
    B, H, W = depth.shape
    assert intri.shape == (B, 3, 3)
    assert extri.shape == (B, 4, 4)
    # get the focal length
    f = intri[..., 0, 0]
    # get the baseline
    if other_extri is None:
        other_extri = torch.zeros_like(extri, dtype=extri.dtype, device=extri.device)
    b = torch.norm(extri[..., :3, 3] - other_extri[..., :3, 3])
    # b = extri[:, 0, 3]
    # get the disparity
    shape = f.shape + (1,)  * (depth.ndim - f.ndim)
    disparity = f.view(shape) * b.view(shape) / depth
    return disparity

def train(args):
    expdir = args.expdir
    ckptdir = os.path.join(expdir, 'ckpts')
    logdir = os.path.join(expdir, 'tensorboard')
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    model = TriRAFTStereo(args)
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth.gz"), f"invalid ckpt path: {args.restore_ckpt}"
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")
    if args.continual_train:
        from glob import glob
        ckpt_file_paths = glob(os.path.join(ckptdir, "*_epoch_raft-stereo.pth.gz"))
        ckpt_file_paths = sorted(ckpt_file_paths,key=lambda x: int(os.path.basename(x).split("_")[0]))
        ckpt_to_load = ckpt_file_paths[-1]
        ckpt = torch.load(ckpt_to_load)
        model.load_state_dict(ckpt, strict=True)
        logging.info(f"Load checkpoint from {ckpt_to_load}")

    model = nn.DataParallel(model)
    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_tri_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler, logdir)

    model.cuda()
    model.train()
    model.module.freeze_bn() # We keep BatchNorm frozen

    validation_frequency = 10000

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    epoch = 0
    max_grad_norm = args.max_grad_norm if hasattr(args, 'max_grad_norm') else 1.
    while should_keep_training:
        data_iter = tqdm(train_loader, total=len(train_loader) if args.steps_per_epoch is None else args.steps_per_epoch)
        for i_batch, data_blob in enumerate(data_iter): # for original datasets
        # for i_batch, idict in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            # image1, image2, flow, valid = [x.cuda() for x in data_blob]# for original datasets
            
            image_l, image_r, image_m, flow, valid = [x.cuda() for x in data_blob]
            
            # # debug
            # print(image_l.shape, image_r.shape, image_m.shape, flow.shape, valid.shape)
            # import cv2
            # import matplotlib.pyplot as plt
            # os.makedirs("./debug", exist_ok=True)
            # for i in range(image_l.shape[0]):
            #     limg = image_l[i].permute(1,2,0).cpu().numpy().astype(np.uint8)
            #     rimg = image_r[i].permute(1,2,0).cpu().numpy().astype(np.uint8)
            #     mimg = image_m[i].permute(1,2,0).cpu().numpy().astype(np.uint8)
            #     disp = flow[i].squeeze().cpu().abs().numpy()
            #     cv2.imwrite(os.path.join("debug", f"L_img_{i}.png"), limg)
            #     cv2.imwrite(os.path.join("debug", f"R_img_{i}.png"), rimg)
            #     cv2.imwrite(os.path.join("debug", f"M_img_{i}.png"), mimg)
            #     plt.imshow(disp, cmap='jet', vmin=disp.min(), vmax=disp.max())
            #     plt.colorbar()
            #     plt.savefig(os.path.join("debug", f"disp_{i}.png"))
            #     plt.close()
            # # 记得注释：debug

            assert model.training
            flow_predictions = model(image_l, image_r, image_m, iters=args.train_iters)
            assert model.training
            loss, metrics = sequence_loss(flow_predictions, flow, valid)
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            logger.push(metrics)

            # 取消validation
            # if total_steps % validation_frequency == validation_frequency - 1:
            #     save_path = Path(os.path.join(ckptdir, '%d_%s.pth' % (total_steps + 1, args.name)))
            #     logging.info(f"Saving file {save_path.absolute()}")
            #     torch.save(model.state_dict(), save_path)

            #     results = validate_things(model.module, iters=args.valid_iters)

            #     logger.write_dict(results)

            #     model.train()
            #     model.module.freeze_bn()

            total_steps += 1
            # mem usage
            memory_percent = psutil.virtual_memory().percent
            rss = psutil.Process().memory_info().rss / 1024.**2
            data_iter.set_description(f"loss: {loss.item():.3f};mem:{memory_percent:.3f}%,rss:{rss:.2f}MB")
            # data_iter.set_description(f"mem:{memory_percent:.3f}%,rss:{rss:.2f}MB")

            if total_steps > args.num_steps:
                should_keep_training = False
                break

            # if len(train_loader) >= 10000:
            if total_steps % 10000 == 0:
                save_path = Path(os.path.join(ckptdir, '%d_steps_%s.pth.gz' % (total_steps + 1, args.name)))
                logging.info(f"Saving file {save_path}")
                torch.save(model.module.state_dict(), save_path)

            if args.steps_per_epoch is not None and total_steps % args.steps_per_epoch == 0:
                break
            # break  # DEBUG
        
        epoch += 1
        save_path = Path(os.path.join(ckptdir, '%d_epoch_%s.pth.gz' % (epoch, args.name)))
        logging.info(f"Saving file {save_path}")
        torch.save(model.module.state_dict(), save_path)
        # break  # DEBUG


    print("FINISHED TRAINING")
    logger.close()
    PATH = os.path.join(ckptdir, '%s.pth' % args.name)
    torch.save(model.module.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    '''
    default command: 
    python train_stereo.py --batch_size 8 --train_iters 22 --valid_iters 32 \
        --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 \
        --num_steps 200000 --mixed_precision
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--expdir", type=str, required=True)
    parser.add_argument("--deepsl_args_path", default="./cfgs/train_local_matchlr.yaml", type=str)
    parser.add_argument('--name', default='raft-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--continual_train', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--steps_per_epoch', type=int, default=None)

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['dataset_deepsl'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
    parser.add_argument('--no_gray', action='store_true')

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--shared_fnet', type=bool, default=False)
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--corr_multiplier', type=int, default=2)
    parser.add_argument('--corr_middle_rate', type=float, default=0.5)
    parser.add_argument('--buggy_tri_corr', type=bool, default=False)
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    # Path("checkpoints").mkdir(exist_ok=True, parents=True)

    train(args)