import os
import random
import sys
from datetime import datetime
from functools import reduce

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader  # new add

import util.utils as util
from config.default_config import DefaultConfig
from config.segmentation.resnet_sunrgbd_config import RESNET_SUNRGBD_CONFIG
from config.segmentation.resnet_cityscape_config import RESNET_CITYSCAPE_CONFIG
from config.resnet_stl10_config import RESNET_STL10_CONFIG

from data import segmentation_dataset_cv2
from data import segmentation_dataset
from data import stl10_dataset

from model.trans2_model import Trans2Net
from model.trans2_multimodal import TRans2Multimodal

import torch.multiprocessing as mp
import torch.distributed as dist
import apex
import cv2
import torch.nn as nn

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def main():
    cfg = DefaultConfig()
    args = {
        'resnet_sunrgbd': RESNET_SUNRGBD_CONFIG().args(),
        'resnet_cityscapes': RESNET_CITYSCAPE_CONFIG().args(),
        'resnet_stl10': RESNET_STL10_CONFIG().args()

    }

    # use shell
    if len(sys.argv) > 1:
        device_ids = torch.cuda.device_count()
        print('device_ids:', device_ids)
        gpu_ids, config_key = sys.argv[1:]
        cfg.parse(args[config_key])
        cfg.GPU_IDS = gpu_ids.split(',')

    else:
        # config_key = 'resnet_sunrgbd'
        # config_key = 'resnet_cityscapes'
        config_key = 'resnet_stl10'
        cfg.parse(args[config_key])
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(lambda x: str(x), cfg.GPU_IDS))

    trans_task = ''
    cfg.NO_TRANS=True
    # cfg.S
    cfg.MULTIPROCESSING_DISTRIBUTED=True
    # if not cfg.NO_TRANS:
    #     if cfg.MULTI_MODAL:
    #         trans_task = trans_task + '_multimodal'
    #     else:
    #         trans_task = trans_task + '_to_' + cfg.TARGET_MODAL

    #     trans_task = trans_task + '_alpha_' + str(cfg.ALPHA_CONTENT)

    # evaluate_type = 'sliding_window' if cfg.SLIDE_WINDOWS else 'center_crop'

    # cfg.LOG_PATH = os.path.join(cfg.LOG_PATH, cfg.MODEL, cfg.CONTENT_PRETRAINED,
    #                             ''.join(
    #                                 [cfg.TASK, trans_task, '_', cfg.DATASET, '_', '.'.join(cfg.LOSS_TYPES), '_',
    #                                  evaluate_type,
    #                                  '_gpus_', str(len(cfg.GPU_IDS))]), datetime.now().strftime('%b%d_%H-%M-%S'))

    # Setting random seed
    if cfg.MANUAL_SEED is None:
        cfg.MANUAL_SEED = random.randint(1, 10000)
    random.seed(cfg.MANUAL_SEED)
    torch.manual_seed(cfg.MANUAL_SEED)

    torch.backends.cudnn.benchmark = True

    project_name = reduce(lambda x, y: str(x) + '/' + str(y), os.path.realpath(__file__).split(os.sep)[:-1])
    print('>>> task path is {0}'.format(project_name))

    util.mkdir('logs')

    # dataset = segmentation_dataset
    dataset = stl10_dataset
    # dataset = segmentation_dataset_cv2
    which_dataset = None
    train_transforms = list()
    val_transforms = list()
    ms_targets = []

    if 'sunrgbd' in config_key:
        which_dataset = 'SUNRGBD'
        train_transforms.append(dataset.Resize(cfg.LOAD_SIZE))
        train_transforms.append(dataset.RandomCrop(cfg.FINE_SIZE))
        train_transforms.append(dataset.RandomHorizontalFlip())
        # if cfg.MULTI_SCALE:
        #     for item in cfg.MULTI_SCALE_TARGETS:
        #         ms_targets.append(item)
        #     train_transforms.append(dataset.MultiScale(size=(cfg.FINE_SIZE, cfg.FINE_SIZE),
        #                                                             scale_times=cfg.MULTI_SCALE_NUM, ms_targets=ms_targets))
        train_transforms.append(dataset.ToTensor(ms_targets=ms_targets))
        train_transforms.append(
            dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], ms_targets=ms_targets))

        val_transforms = list()
        val_transforms.append(dataset.Resize(size=(cfg.LOAD_SIZE)))
        if not cfg.SLIDE_WINDOWS:
            val_transforms.append(dataset.CenterCrop((cfg.FINE_SIZE)))
        val_transforms.append(dataset.ToTensor())
        val_transforms.append(dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    elif 'cityscapes' in config_key:

        which_dataset = 'CityScapes'

        # train_transforms.append(dataset.Resize(cfg.LOAD_SIZE))
        train_transforms.append(dataset.RandomScale(cfg.RANDOM_SCALE_SIZE))
        train_transforms.append(dataset.RandomCrop(cfg.FINE_SIZE))
        # train_transforms.append(dataset.RandomRotate())
        train_transforms.append(dataset.RandomHorizontalFlip())
        train_transforms.append(dataset.ToTensor(ms_targets=ms_targets))
        train_transforms.append(
            dataset.Normalize(mean=cfg.MEAN, std=cfg.STD, ms_targets=ms_targets))

        # val_transforms.append(dataset.Resize(cfg.LOAD_SIZE))
        if not cfg.SLIDE_WINDOWS:
            val_transforms.append(dataset.CenterCrop((cfg.FINE_SIZE)))
        val_transforms.append(dataset.ToTensor())
        val_transforms.append(dataset.Normalize(mean=cfg.MEAN, std=cfg.STD))

    elif 'ade20k' in config_key:

        which_dataset = 'ADE20K'

        train_transforms.append(dataset.Resize(cfg.LOAD_SIZE))
        train_transforms.append(dataset.RandomScale(cfg.RANDOM_SCALE_SIZE))  #
        train_transforms.append(dataset.RandomCrop(cfg.FINE_SIZE))  #
        # train_transforms.append(dataset.RandomRotate())
        train_transforms.append(dataset.RandomHorizontalFlip())
        train_transforms.append(dataset.ToTensor(ms_targets=ms_targets))
        train_transforms.append(
            dataset.Normalize(mean=cfg.MEAN, std=cfg.STD, ms_targets=ms_targets))

        val_transforms.append(dataset.Resize(cfg.LOAD_SIZE))
        # val_transforms.append(dataset.CenterCrop(cfg.FINE_SIZE))
        val_transforms.append(dataset.ToTensor())
        val_transforms.append(dataset.Normalize(mean=cfg.MEAN, std=cfg.STD))
    elif 'stl10' in config_key:

        which_dataset = 'SPL10_Dataset'


        train_transforms.append(dataset.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE)))
        train_transforms.append(dataset.RandomCrop((cfg.FINE_SIZE, cfg.FINE_SIZE)))
        train_transforms.append(dataset.RandomHorizontalFlip())
        if cfg.MULTI_SCALE:
            train_transforms.append(dataset.MultiScale((cfg.FINE_SIZE, cfg.FINE_SIZE), scale_times=cfg.MULTI_SCALE_NUM))
        train_transforms.append(dataset.RGB2Lab()),
        train_transforms.append(dataset.ToTensor())
        train_transforms.append(dataset.Normalize(mean = [0.485, 0.456, 0.406],
                                                  std = [0.229, 0.224, 0.225]))


        val_transforms.append(dataset.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE))),
        val_transforms.append(dataset.CenterCrop((cfg.FINE_SIZE, cfg.FINE_SIZE))),
        val_transforms.append(dataset.RGB2Lab()),
        val_transforms.append(dataset.ToTensor()),
        val_transforms.append(dataset.Normalize(mean = [0.485, 0.456, 0.406],
                                                  std = [0.229, 0.224, 0.225]))



    train_dataset = dataset.__dict__[which_dataset](cfg=cfg, transform=transforms.Compose(train_transforms),
                                                    # phase_train=True,
                                                    data_dir=cfg.DATA_DIR_TRAIN)
    val_dataset = dataset.__dict__[which_dataset](cfg=cfg, transform=transforms.Compose(val_transforms), 
                                                    # phase_train=False,
                                                  data_dir=cfg.DATA_DIR_VAL)
    # cfg.CLASS_WEIGHTS_TRAIN = train_dataset.class_weights
    cfg.IGNORE_LABEL = 255

    cfg.train_dataset = train_dataset
    cfg.val_dataset = val_dataset
    cfg.MULTIPROCESSING_DISTRIBUTED=True
    if cfg.MULTIPROCESSING_DISTRIBUTED:
        cfg.rank = 0
        cfg.ngpus_per_node = len(cfg.GPU_IDS)
        cfg.dist_url = 'tcp://127.0.0.1:8888'
        cfg.dist_backend = 'nccl'
        cfg.opt_level = 'O0'
        cfg.world_size = 1

    cfg.print_args()

    ngpus_per_node = len(cfg.GPU_IDS)
    if cfg.MULTIPROCESSING_DISTRIBUTED:
        cfg.world_size = ngpus_per_node * cfg.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        # Simply call main_worker function
        main_worker(cfg.GPU_IDS, ngpus_per_node, cfg)


def main_worker(gpu, ngpus_per_node, cfg):

    writer = SummaryWriter(log_dir=cfg.LOG_PATH)  # tensorboard

    if cfg.MULTIPROCESSING_DISTRIBUTED:
        cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url, world_size=cfg.world_size,
                                rank=cfg.rank)
        torch.cuda.set_device(gpu)

        # data
        cfg.BATCH_SIZE_TRAIN = int(cfg.BATCH_SIZE_TRAIN / ngpus_per_node)
        cfg.BATCH_SIZE_VAL = int(cfg.BATCH_SIZE_VAL / ngpus_per_node)
        cfg.WORKERS = int(cfg.WORKERS / ngpus_per_node)
        train_sampler = torch.utils.data.distributed.DistributedSampler(cfg.train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(cfg.val_dataset)
        cfg.train_sampler = train_sampler
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(cfg.train_dataset, batch_size=cfg.BATCH_SIZE_TRAIN, shuffle=(train_sampler is None),
                              num_workers=cfg.WORKERS, pin_memory=True, drop_last=True, sampler=train_sampler)

    val_loader = DataLoader(cfg.val_dataset, batch_size=cfg.BATCH_SIZE_VAL, shuffle=False,
                            num_workers=cfg.WORKERS, pin_memory=True, sampler=val_sampler)

    # model
    if cfg.SYNC_BN:
        if cfg.MULTIPROCESSING_DISTRIBUTED:
            BatchNorm = apex.parallel.SyncBatchNorm
        else:
            from lib.sync_bn.modules import BatchNorm2d as SyncBatchNorm
            BatchNorm = SyncBatchNorm
    else:
        BatchNorm = nn.BatchNorm2d

    if cfg.MULTI_MODAL:
        model = TRans2Multimodal(cfg, writer=writer)
    else:
        model = Trans2Net(cfg, writer=writer, batch_norm=BatchNorm)
    model.set_data_loader(train_loader, val_loader)

    if cfg.RESUME:
        checkpoint_path = os.path.join(cfg.CHECKPOINTS_DIR, cfg.RESUME_PATH)
        model.load_checkpoint(model.net, checkpoint_path)
        if cfg.INIT_EPOCH:
            # just load pretrained parameters
            print('load checkpoint from another source')
            cfg.START_EPOCH = 1

    # train
    model.train_parameters(cfg)

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
