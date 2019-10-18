import os
import shutil
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler


class BaseModel(nn.Module):

    def name(self):
        pass

    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.gpu_ids = cfg.GPU_IDS
        self.model = None
        self.device = torch.device('cuda' if self.gpu_ids else 'cpu')
        self.save_dir = os.path.join(self.cfg.CHECKPOINTS_DIR, self.cfg.MODEL, str(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))))
        if not os.path.exists(self.save_dir):
            # shutil.rmtree(self.save_dir)
            os.mkdir(self.save_dir)
        # else:
        #     os.mkdir(self.save_dir)


    # schedule for modifying learning rate
    def set_schedulers(self, cfg):

        self.schedulers = [self._get_scheduler(optimizer, cfg, cfg.LR_POLICY) for optimizer in self.optimizers]

    def _get_scheduler(self, optimizer, cfg, lr_policy):
        if lr_policy == 'lambda':
            print('use lambda lr')
            decay_start = cfg.NITER
            decay_iters = cfg.NITER_DECAY
            total_iters = cfg.NITER_TOTAL

            # assert NITER_TOTAL == decay_start + decay_iters

            def lambda_rule(iters):
                # lr_l = (1 - float(iters) / total_iters) ** 0.9

                lr_l = 1 - max(0, iters - decay_start) / float(decay_iters)
                if lr_l < 1:
                    lr_l = lr_l
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif lr_policy == 'step':
            print('use step lr')
            scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.LR_DECAY_ITERS, gamma=0.1)
        elif lr_policy == 'plateau':
            print('use plateau lr')
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True,
                                                       threshold=0.0001, factor=0.5, patience=2, eps=1e-7)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
        return scheduler

    def set_data_loader(self, train_loader=None, val_loader=None):

        if train_loader is not None:
            self.train_loader = train_loader
            self.train_image_num = self.train_loader.dataset.__len__()

            print('train_num:',self.train_image_num)
        if val_loader is not None:
            self.val_loader = val_loader
            self.val_image_num = self.val_loader.dataset.__len__()
            print('val_num:', self.val_image_num)

    def load_checkpoint(self, net, checkpoint_path):

        if os.path.isfile(checkpoint_path):

            state_checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
            print('loading model...')
            state_dict = net.state_dict()
            state_checkpoint = {str.replace(k, 'module.', ''): v for k, v in state_checkpoint['state_dict'].items()}

            state_dict.update(state_checkpoint)
            net.load_state_dict(state_dict)

        else:
            print("=> !!! No checkpoint found at '{}'".format(self.cfg.RESUME))
            return

    def save_checkpoint(self, iter, filename=None):

        if filename is None:
            filename = 'Trans2_{0}_{1}.pth'.format(self.cfg.MODEL, iter)

        net_state_dict = self.net.state_dict()
        save_state_dict = {}
        for k, v in net_state_dict.items():
            if 'content_model' in k:
                continue
            save_state_dict[k] = v

        state = {
            'iter': iter,
            'state_dict': save_state_dict,
            'optimizer': self.optimizer.state_dict(),
        }

        filepath = os.path.join(self.save_dir, filename)
        torch.save(state, filepath)

    def update_learning_rate(self, val=None, step=None):
        for scheduler in self.schedulers:
            if val is not None:
                scheduler.step(val)
            else:
                scheduler.step(step)

    def print_lr(self):
        for optimizer in self.optimizers:
            # print('default lr', optimizer.defaults['lr'])
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                print('/////////learning rate = %.7f' % lr)

    def set_log_data(self, cfg):
        pass


