import os
import numpy as np
import torch
import torch.multiprocessing.spawn
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from utils.common import FileIOHelper, copy_file, check_make_dirs, to_device, manual_seed
from utils.dist import setup_ddp, get_free_port, print_ddp, barrier, setup_distributed_full
from utils.args import ParserWrapper
from utils.models import parse_optimizer, parse_scheduler
from utils.aug import parse_augmentation
from utils.loss import parse_losses
from utils.transforms import interp_disp_to_depth
from utils.visualize import vis_batch
from deepsl_data.dataloader.dataloader import create_simplified_stereo_dataloader_with_pattern
from zoo.neuralsl.sl_prompt_da import create_model

# # DEBUG
# def check_nan_grad(model:nn.Module):
#     nan_keys = []
#     for k, v in model.named_parameters():
#         if v.grad is not None and (v.grad.isnan().any() or v.grad.isinf().any()):
#             nan_keys.append(k)
#         if k == 'module.depth_head.scratch.refinenet1.prompt_feat_projector.0.weight':
#             print_ddp(f"grad module.depth_head.scratch.refinenet1.prompt_feat_projector.0.weight: {v.min().item(), v.max().item(), v.shape}")
#     return nan_keys

# # DEBUG
# def check_optimizer_state(optimizer):
#     for group in optimizer.param_groups:
#         for p in group['params']:
#             if p.grad is not None:
#                 state = optimizer.state[p]
#                 # 检查优化器状态中是否有 NaN 或 Inf
#                 for key, value in state.items():
#                     if isinstance(value, torch.Tensor) and (torch.isnan(value).any() or torch.isinf(value).any()):
#                         print(f"Before step - Optimizer state '{key}' contains NaN or Inf!")

def check_nan_parameters(model:nn.Module):
    nan_keys = []
    for k, v in model.named_parameters():
        if v.isnan().any():
            nan_keys.append(k)
    assert len(nan_keys) == 0, f"Encounter NaN parameters, stop training! {len(nan_keys)} NaN: {'; '.join(nan_keys)}"

def parse_args():
    parser = ParserWrapper()
    parser.add(ParserWrapper.ARGS_GROUP_TRAIN)  # dataloader contained in cfgs.
    parser.add(ParserWrapper.ARGS_GROUP_EXP)
    parser.add_argument('--local-rank', default=0, type=int)
    args = parser.parse_args()
    return args

def train_extra_loss(extra_info:dict, gt_depth:torch.Tensor, extra_loss_cfg:dict):
    if not hasattr(train_extra_loss, 'loss_module'):
        train_extra_loss.loss_module = parse_losses(**extra_loss_cfg.loss)
    exloss = {}
    dep_from_pat = extra_info[0]
    exloss = train_extra_loss.loss_module.forward(
        dep_from_pat['depth'], gt_depth, None, dep_from_pat['conf']
    )
    # print(dep_from_pat['depth'].min().item(), dep_from_pat['depth'].max().item(),
    #       dep_from_pat['conf'].min().item(),dep_from_pat['conf'].max().item())
    if len(extra_info) == 2:  # stereo setup.
        dep_from_mv = extra_info[1]
        gamma = extra_loss_cfg.get("gamma", 0.9)
        for i, ex in enumerate(dep_from_mv):
            # print(ex['depth'].min().item(),ex['depth'].max().item(),
            #       ex['conf'].min().item(),ex['conf'].max().item())
            layerid = len(dep_from_mv) - i
            coef = gamma ** (layerid - 1)
            l = train_extra_loss.loss_module.forward(
                ex['depth'], gt_depth, None, ex['conf']
            )
            tot = l.pop("total_loss")
            exloss['total_loss'] = tot * coef
            for k, v in l.items():
                exloss[f"layer_{layerid}_{k}"] = v
    return exloss

def train_one_epoch(
        rank:int, world_size:int, train_cfgs:dict, epoch_id:int,
        model:DDP|nn.Module, dataloader:DataLoader, loss:nn.Module,
        optimizer:Optimizer, lr_scheduler:LRScheduler, aug,
        writer:SummaryWriter, start_steps:int, num_steps:int = None,
        monitor_mem:bool = False
    ):
    if monitor_mem:
        import psutil
    device = torch.device(f"cuda:{rank}")
    near = train_cfgs.get('near', train_cfgs.get('min_depth', 0.1))
    far = train_cfgs.get('far', train_cfgs.max_depth)
    steps = start_steps
    num_steps = num_steps if num_steps is not None else len(dataloader)
    data_iterator = tqdm(dataloader, total=num_steps, desc=f"epoch {epoch_id}") \
                    if rank == 0 else iter(dataloader)
    extra_loss_cfg = train_cfgs.get("extra_loss", None)
    # if train_cfgs.get("keep_modelout", False):
    #     modelout_depth_type = 'nochange'
    # else:
    #     modelout_depth_type = train_cfgs.get("depth_type", "norm_depth")
    for sample in data_iterator:
        sample['near'] = near
        sample['far'] = far
        sample = aug(to_device(sample, device))
        kwargs = train_cfgs.get("fwd_kwargs", {})
        l_depth, r_depth, *extra_info = model.forward(**{**sample, **kwargs})
        pred_depth = torch.concat((l_depth, r_depth), dim=0) if r_depth is not None else l_depth
        gt_depth = torch.concat((sample['L_Depth'], sample['R_Depth']), dim=0) if r_depth is not None else sample['L_Depth']
        loss_dict = loss(pred_depth, gt_depth)

        # loss of extra info...
        if extra_loss_cfg is not None and len(extra_info) != 0:
            extra_loss_dict = train_extra_loss(extra_info, gt_depth, extra_loss_cfg)
            tot = extra_loss_dict.pop("total_loss")
            loss_dict['total_loss'] = loss_dict['total_loss'] + tot * extra_loss_cfg['weight']
            for k, v in extra_loss_dict.items():
                loss_dict[k] = v
            del extra_loss_dict
        
        # update
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        if train_cfgs.get("grad_norm", None) is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfgs['grad_norm'])
        optimizer.step()
        lr_scheduler.step()
        # log
        if rank == 0:
            loss_str = ",".join([f"{k}={v.item():.2e}" for k, v in loss_dict.items()])
            desc = f"epoch-{epoch_id},lr={optimizer.param_groups[0]['lr']:.2e}," + loss_str
            if monitor_mem:
                memory_percent = psutil.virtual_memory().percent
                desc += f";mem:{memory_percent:.3f}%"
                if train_cfgs.get("grad_norm", None) is not None:
                    desc += f";gradnorm:{grad_norm.item():.3f}"
            data_iterator.set_description(desc)
            for k, v in loss_dict.items():
                writer.add_scalar(k, v.item(), global_step=steps)
        steps += 1

        if steps % 1000 == 0 and train_cfgs.get('ckptdir', None) is not None:
            check_nan_parameters(model)
            if rank == 0:
                ckpt = {
                    'model': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch_id, 'steps': steps
                }
                iohelper = FileIOHelper()
                with iohelper.open(os.path.join(train_cfgs['ckptdir'], 'latest.pt'), 'wb') as f:
                    torch.save(ckpt, f)
            barrier()
        # if True:       # Debug
        if "n_steps_per_vis" in train_cfgs and steps % train_cfgs["n_steps_per_vis"] == 0:
            # TODO: 修改成更统一的depth_coversion函数.
            depth_type = train_cfgs.get("depth_type", "norm_depth")
            if depth_type.endswith('disp'):
                with torch.no_grad():
                    sample['L_Depth'] = interp_disp_to_depth(sample['L_Depth'], sample['near'], sample['far'], sample['L_intri'],
                                                             sample['L_extri'], sample['P_extri'], depth_type)
                    sample['R_Depth'] = interp_disp_to_depth(sample['R_Depth'], sample['near'], sample['far'], sample['R_intri'],
                                                             sample['R_extri'], sample['P_extri'], depth_type)
                    l_depth = interp_disp_to_depth(l_depth, sample['near'], sample['far'], sample['L_intri'],
                        sample['L_extri'], sample['P_extri'], depth_type
                    )
                    if r_depth is not None:
                        r_depth = interp_disp_to_depth(r_depth, sample['near'], sample['far'], sample['R_intri'],
                            sample['R_extri'], sample['P_extri'], depth_type
                        )
            vis_batch(sample, l_depth, r_depth, steps, train_cfgs.expdir, train_cfgs.get('max_depth', 10), rank, train_cfgs.get("loss_range", [-0.1, 0.1]))
            torch.cuda.empty_cache()
            barrier()
        if steps - start_steps >= num_steps:
            break
        # if steps >= 2:  # debug
        #     exit(0)  # debug

def train(
        rank, world_size, args:dict, port, ddp:bool=True, init_ddp:bool=True
    ):
    # ###### only for debug #######
    # args.cfgs.Train.n_steps_per_vis = 1
    # args.cfgs.Train.n_epoches_per_ckpt
    # args.cfgs.Train.n_epoches = 1
    # args.cfgs.Val.n_steps_per_vis = 1
    # #############################
    iohelper = FileIOHelper()
    # init ddp.
    if ddp and init_ddp:
        setup_ddp(port, rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    # create models.
    cfgs = args.cfgs
    model_cfgs = cfgs.Model
    model, ckpt = create_model(return_ckpt=True, **model_cfgs)  #.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.extra_setup_()
    model = model.to(device)
    print_ddp("Model created.")
    # continue train.
    start_epoch = 0
    continue_steps = 0
    if model_cfgs.get('ckpt_path', None) is not None and 'epoch' in ckpt:  # continue train
        ckpt_fname = os.path.basename(model_cfgs['ckpt_path'])
        if ckpt_fname == 'latest.pt':
            start_epoch = ckpt['epoch']  # do not further consider which step it is within the epoch. 
            # ckpt['steps'] // num_steps_per_epoch may be better?
            continue_steps = ckpt['steps']
        else:
            start_epoch = int(ckpt_fname.split("_")[0]) + 1
            continue_steps = 0
    
    # create optimizer
    train_cfgs = cfgs.Train
    optimizer:Optimizer = parse_optimizer(model, **train_cfgs.optimizer)
    if ckpt is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
        print_ddp("optimizer ckpt loaded.")
    # create dataloader
    dataset_cfgs = cfgs.Dataset
    dataset_name = dataset_cfgs.get("dataset_name", 'deepsl')
    if 'dataset_name' in dataset_cfgs:
        dataset_cfgs.pop('dataset_name')
    if dataset_name == 'deepsl':
        dataloader = create_simplified_stereo_dataloader_with_pattern(
            rank=rank, world_size=world_size, ddp=ddp, **dataset_cfgs
        )
    elif dataset_name == 'dreds':
        from datasets.dreds import create_dreds_dataloader
        dataloader = create_dreds_dataloader(
            rank=rank, world_size=world_size, ddp=ddp, **dataset_cfgs
        )
    else:
        raise NotImplementedError
    # create scheduler
    scheduler:LRScheduler = parse_scheduler(optimizer, len(dataloader), **train_cfgs.schedulers)
    if ckpt is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
        print_ddp("scheduler ckpt loaded")
    # create loss module
    loss = parse_losses(**train_cfgs.losses)
    # create augmentation module.
    aug = parse_augmentation(*train_cfgs.aug)
    print_ddp("Optimizer, LRscheduler, augmentation, loss modules created.")
    print_ddp("Dataloader created.")

    # if master, create summarywriter.
    print_ddp(f"summary writer log dir: {args.logdir}")
    summary_writer = SummaryWriter(log_dir=args.logdir) if rank == 0 else None

    if ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        print_ddp("DDP created.")
        if ckpt is not None:
            model.module.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
            print_ddp("model ckpt loaded.")

    # train loop.
    num_steps_per_epoch = train_cfgs.get("n_steps_per_epoch", len(dataloader))
    train_cfgs.ckptdir = args.ckptdir
    print_ddp(f"Start training at epoch: {start_epoch}; start steps for continuous training: {continue_steps}")
    for epoch_id in range(start_epoch, train_cfgs.n_epoches):
        start_steps = num_steps_per_epoch * epoch_id + (continue_steps if epoch_id == start_epoch else 0)
        dataloader.sampler.set_epoch(epoch_id)
        model.train()
        train_one_epoch(rank, world_size, train_cfgs, epoch_id,
            model, dataloader, loss, optimizer, scheduler,
            aug, summary_writer, start_steps, num_steps_per_epoch,
            monitor_mem=True
        )
        barrier()
        # model.eval()   节省时间不再eval，而是之后手动inference...
        # with torch.no_grad():
        #     # validation
        #     metric_dict = validation_given_model(rank, world_size, model, args, val_on_test=True, ddp=True)  # 按照workder划分快一点。。。
        # barrier()

        # dump ckpts
        check_nan_parameters(model)
        if rank == 0 and (epoch_id + 1) % train_cfgs.n_epoches_per_ckpt == 0:
            model_state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            optim_state_dict = optimizer.state_dict()
            sche_state_dict = scheduler.state_dict()
            ckpt = {
                'epoch': epoch_id,
                'model': model_state_dict,
                'optimizer': optim_state_dict,
                'scheduler': sche_state_dict,
            }
            # keys = sorted(list(metric_dict.keys()))
            # metric_str = [f"{k}-{metric_dict[k]:.2f}" for k in keys]
            # metric_str = "_".join(metric_str)
            # ckpt_name = f"{epoch_id:02d}_{metric_str}.pt"
            ckpt_name = f"{epoch_id:02d}.pt"
            with iohelper.open(os.path.join(args.ckptdir, ckpt_name), 'wb') as f:
                torch.save(ckpt, f)
        barrier()


if __name__ == '__main__':
    import logging
    import json
    import os
    
    dist_inited = False
    if 'RANK' in os.environ:
        rank, world_size = setup_distributed_full()
        dist_inited = True

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

    args = parse_args()
    # seed all
    if args.cfgs.Train.get("seed", None) is not None:
        seed = args.cfgs.Train.seed
    else:
        seed = args.seed
    manual_seed(seed)
    # exp
    args.cfgs.Train.expdir = expdir = args.expdir
    if args.logdir is None:
        args.logdir = os.path.join(expdir, 'logs')
    if args.ckptdir is None:
        args.ckptdir = os.path.join(expdir, 'ckpt')
    if not dist_inited or rank == 0:
        check_make_dirs(expdir, args.logdir, args.ckptdir)
        ## copy cfgs file.
        copy_file(args.cfgs_path, os.path.join(expdir, os.path.basename(args.cfgs_path)))
        ## dump full args.
        iohelper = FileIOHelper()
        with iohelper.open(os.path.join(expdir, 'cmd_args.json'), 'w') as f:
            json.dump(args, f, indent=4)

    # spawn ddp.
    ngpus = torch.cuda.device_count()
    ngpus = min(ngpus, args.cfgs.Train.n_gpus)
    args.cfgs.Train.n_gpus = ngpus
    port = get_free_port()
    ddp = not args.no_ddp
    
    if not dist_inited or rank==0:
        logger.info(f"cfgs: {args.cfgs_path}")
        logger.info(f"n_gpus: {ngpus}")
        logger.info(f"n_epoches: {args.cfgs.Train.n_epoches}")
        logger.info(f"ddp training: {ddp}")
        logger.info(f"DDP starts.")

    if ddp and not dist_inited:
        torch.multiprocessing.spawn(
            train, nprocs=ngpus, args=(ngpus, args, port)
        )
    elif dist_inited:
        train(rank, world_size, args, port, True, False)
    else:
        train(0, 1, args, port, False)