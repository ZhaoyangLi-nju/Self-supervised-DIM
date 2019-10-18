import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision

import util.utils as util
from util.average_meter import AverageMeter
from . import networks as networks
from .base_model import BaseModel
from tqdm import tqdm
import cv2
import copy
import apex
import torch.distributed as dist


class Trans2Net(BaseModel):

    def __init__(self, cfg, writer=None, batch_norm=nn.BatchNorm2d):
        super(Trans2Net, self).__init__(cfg)

        self.phase = cfg.PHASE
        self.trans = not cfg.NO_TRANS
        self.content_model = None
        self.writer = writer
        self.batch_size_train = cfg.BATCH_SIZE_TRAIN
        self.batch_size_val = cfg.BATCH_SIZE_VAL

        # networks
        # networks.batch_norm = batch_norm
        self.net = networks.define_netowrks(cfg, device=self.device,Batch_norm=batch_norm)
        self.net = self.net.to(self.device)
        # networks.print_network(self.net)

        if 'PSP' in cfg.MODEL:
            self.modules_ori = [self.net.layer0, self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
            self.modules_new = [self.net.ppm, self.net.cls, self.net.aux]

            if self.trans:
                self.modules_new.extend([self.net.up1, self.net.up2, self.net.up3,
                                         self.net.up4, self.net.up_seg])
            self.params_list = []
            for module in self.modules_new:
                self.params_list.append(dict(params=module.parameters(), lr=cfg.LR * 10))
            for module in self.modules_ori:
                self.params_list.append(dict(params=module.parameters(), lr=cfg.LR))
        
        if cfg.USE_FAKE_DATA:
            print('Use fake data: sample model is {0}'.format(cfg.SAMPLE_MODEL_PATH))
            print('fake ratio:', cfg.FAKE_DATA_RATE)
            sample_model_path = cfg.SAMPLE_MODEL_PATH
            cfg_sample = copy.deepcopy(cfg)
            cfg_sample.USE_FAKE_DATA = False
            model = networks.define_netowrks(cfg_sample, device=self.device)
            self.load_checkpoint(net=model, checkpoint_path=sample_model_path)
            model.eval()
            self.sample_model = nn.DataParallel(model).to(self.device)

    def _optimize(self):

        self._forward()
        self.optimizer.zero_grad()
        total_loss = self._construct_loss()

        if self.cfg.USE_APEX and self.cfg.MULTIPROCESSING_DISTRIBUTED:
            with apex.amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()

        self.optimizer.step()

    def set_criterion(self, cfg):

        if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.EVALUATE:
            criterion_segmentation = util.CrossEntropyLoss2d(weight=cfg.CLASS_WEIGHTS_TRAIN,
                                                             ignore_index=cfg.IGNORE_LABEL)
            self.net.set_cls_criterion(criterion_segmentation)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES:
            criterion_content = torch.nn.L1Loss()
            content_model = networks.Content_Model(cfg, criterion_content).to(self.device)
            self.net.set_content_model(content_model)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES:
            criterion_pix2pix = torch.nn.L1Loss()
            self.net.set_pix2pix_criterion(criterion_pix2pix)

    def set_input(self, data):

        self._source = data['image']
        self.source_modal = self._source.to(self.device)
        self.batch_size = self._source.size()[0]
        # if 'label' in data.keys():
        #     self._label = data['label']
        #     self.label = torch.LongTensor(self._label).to(self.device)

        # if self.trans and not self.cfg.MULTI_MODAL:
        #     target_modal = data[self.cfg.TARGET_MODAL]
        #     self.target_modal = target_modal.to(self.device)
        # else:
        #     self.target_modal = None
        self.label = data['label'].to(self.device)
        target_modal = data['lab']
        if isinstance(target_modal, list):
            self.target_modal = list()
            for i, item in enumerate(target_modal):
                self.target_modal.append(item.to(self.device))
        else:
            print('single modal')
            self.target_modal = target.to(self.device)
            # else:
            #     # self.target_modal = util.color_label(self.label)
                
        if self.cfg.WHICH_DIRECTION == 'BtoA':
            self.source_modal, self.target_modal = self.target_modal, self.source_modal

    def train_parameters(self, cfg):

        assert self.cfg.LOSS_TYPES
        # self.set_criterion(cfg)
        self.set_optimizer(cfg)
        self.set_log_data(cfg)
        self.set_schedulers(cfg)
        # self.net = nn.DataParallel(self.net).to(self.device)
        if not cfg.MULTIPROCESSING_DISTRIBUTED:
            self.net = nn.DataParallel(self.net).to(self.device)

        train_total_steps = 0
        train_iters = 0
        best_result = 0

        if self.cfg.EVALUATE and self.cfg.SLIDE_WINDOWS:
            self.prediction_matrix = torch.zeros(self.batch_size_val, self.cfg.NUM_CLASSES, self.cfg.BASE_SIZE[0],
                                            self.cfg.BASE_SIZE[1]).to(self.device)
            self.count_crop_matrix = torch.zeros(self.batch_size_val, 1, self.cfg.BASE_SIZE[0], self.cfg.BASE_SIZE[1]).to(
                self.device)

        # if cfg.INFERENCE:
        #     start_time = time.time()
        #     print('Inferencing model...')
        #     self.evaluate()
        #     print('Evaluation Time: {0} sec'.format(time.time() - start_time))
        #     print(
        #         'MIOU: {miou}, mAcc: {macc}, acc: {acc}'.format(miou=self.loss_meters[
        #                                                                  'VAL_CLS_MEAN_IOU'].val * 100,
        #                                                         macc=self.loss_meters[
        #                                                                  'VAL_CLS_MEAN_ACC'].val * 100,
        #                                                         acc=self.loss_meters[
        #                                                                 'VAL_CLS_ACC'].val * 100))
        #     return

        total_epoch = int(cfg.NITER_TOTAL / math.ceil((self.train_image_num / cfg.BATCH_SIZE_TRAIN)))
        if cfg.MULTIPROCESSING_DISTRIBUTED:
            total_epoch = total_epoch * len(cfg.GPU_IDS)
        print('total epoch:{0}, total iters:{1}'.format(total_epoch, cfg.NITER_TOTAL))

        for epoch in range(cfg.START_EPOCH, total_epoch + 1):

            if train_iters > cfg.NITER_TOTAL:
                break

            if cfg.MULTIPROCESSING_DISTRIBUTED:
                cfg.train_sampler.set_epoch(epoch)

            self.print_lr()

            # current_lr = util.poly_learning_rate(cfg.LR, train_iters, cfg.NITER_TOTAL, power=0.8)

            # if cfg.LR_POLICY != 'plateau':
            #     self.update_learning_rate(step=train_iters)
            # else:
            #     self.update_learning_rate(val=self.loss_meters['VAL_CLS_LOSS'].avg)

            self.fake_image_num = 0
            start_time = time.time()

            self.phase = 'train'
            self.net.train()
            # reset Averagemeters on each epoch
            for key in self.loss_meters:
                self.loss_meters[key].reset()

            iters = 0
            print('gpu_ids:', cfg.GPU_IDS)
            print('# Training images num = {0}'.format(self.train_image_num))
            # batch = tqdm(self.train_loader, total=self.train_image_num // self.batch_size_train)
            # for data in batch:
            for data in self.train_loader:
                self.set_input(data)
                train_iters += 1
                iters += 1
                self._optimize()
                self.update_learning_rate(step=train_iters)

                # self.val_iou = self.validate(train_iters)
                # self.write_loss(phase=self.phase, global_step=train_iters)

            print('log_path:', cfg.LOG_PATH)
            print('iters in one epoch:', iters)
            self.write_loss(phase=self.phase, global_step=train_iters)
            print('Epoch: {epoch}/{total}'.format(epoch=epoch, total=total_epoch))
            util.print_current_errors(util.get_current_errors(self.loss_meters, current=False), epoch)
            print('Training Time: {0} sec'.format(time.time() - start_time))
            if epoch%1==0:
                self.save_best(best_result, epoch, train_iters)
            # if cfg.EVALUATE:
            # if (epoch % 10 == 0 or epoch > total_epoch - 10 or epoch == total_epoch) and cfg.EVALUATE:
            #     print('# Cls val images num = {0}'.format(self.val_image_num))
            #     self.evaluate()
            #     self.print_evaluate_results()
            #     self.write_loss(phase=self.phase, global_step=train_iters)

            #     # save best model
            #     if cfg.SAVE_BEST and epoch > total_epoch - 5:
            #         # save model
            #         self.save_best(best_result, epoch, train_iters)

            print('End of iter {0} / {1} \t '
                  'Time Taken: {2} sec'.format(train_iters, cfg.NITER_TOTAL, time.time() - start_time))
            print('-' * 80)

    def evaluate(self):

        if self.cfg.TASK_TYPE == 'segmentation':
            if not self.cfg.SLIDE_WINDOWS:
                self.validate_seg()
            else:
                self.validate_seg_slide_window()
        elif self.cfg.TASK_TYPE == 'recognition':
            pass

    def save_best(self, best_result, epoch=None, iters=None):
        model_filename = '{0}_{1}_best.pth'.format(self.cfg.MODEL, iters)
        self.save_checkpoint(iters, model_filename)
        print('best is {0}, epoch is {1}, iters {2}'.format(best_result, epoch, iters))
        # if self.cfg.TASK_TYPE == 'segmentation':
        #     miou = self.loss_meters['VAL_CLS_MEAN_IOU'].val
        #     is_best = miou > best_result
        #     best_result = max(miou, best_result)

        #     if is_best:
        #         model_filename = '{0}_{1}_best.pth'.format(self.cfg.MODEL, iters)
        #         self.save_checkpoint(iters, model_filename)
        #         print('best miou is {0}, epoch is {1}, iters {2}'.format(best_result, epoch, iters))

    def print_evaluate_results(self):
        if self.cfg.TASK_TYPE == 'segmentation':
            print(
                'MIOU: {miou}, mAcc: {macc}, acc: {acc}'.format(miou=self.loss_meters[
                                                                         'VAL_CLS_MEAN_IOU'].val * 100,
                                                                macc=self.loss_meters[
                                                                         'VAL_CLS_MEAN_ACC'].val * 100,
                                                                acc=self.loss_meters[
                                                                        'VAL_CLS_ACC'].val * 100))
        elif self.cfg.TASK_TYPE == 'recognition':
            pass

    def _forward(self, cal_loss=True):

        # self.result = self.net(source=self.source_modal, target=self.target_modal, label=self.label, phase=self.phase,
        #                        cal_loss=cal_loss)
        self.result = self.net(source=self.source_modal, target=self.target_modal, label=self.label)

    def _construct_loss(self):

        # loss_total = torch.zeros(1).to(self.device)

        loss_total=None
        if self.cfg.MODEL == 'contrastive':
            l_loss = self.result['gen_l_loss']
            ab_loss = self.result['gen_ab_loss']            
            local_rgb_loss = self.result['local_rgb_loss']
            prior_rgb_loss = self.result['prior_rgb_loss']
            loss_all = l_loss+ab_loss+local_rgb_loss+prior_rgb_loss
            loss_total=loss_all
            dist.all_reduce(loss_all.cuda())
            self.loss_meters['TRAIN_UNLABELED_LOSS'].update(loss_all)

            return loss_total
        # if 'CLS' in self.cfg.LOSS_TYPES:
        #     if self.cfg.MULTIPROCESSING_DISTRIBUTED:
        #         cls_loss = self.result['loss_cls'] * self.cfg.ALPHA_CLS
        #         dist.all_reduce(cls_loss)
        #     else:
        #         cls_loss = self.result['loss_cls'].mean() * self.cfg.ALPHA_CLS
        #     loss_total = cls_loss
        #     cls_loss = round(cls_loss.item(), 4)

        #     self.loss_meters['TRAIN_CLS_LOSS'].update(cls_loss)

        # ) content supervised
        # if 'SEMANTIC' in self.cfg.LOSS_TYPES:

        #     if self.cfg.MULTI_MODAL:
        #         self.gen = [self.result['gen_img_1'], self.result['gen_img_2']]
        #     else:
        #         self.gen = self.result['gen_img']

        #     decay_coef = 1
        #     # decay_coef = (iters / self.cfg.NITER_TOTAL)  # small to big
        #     # decay_coef = max(0, (self.cfg.NITER_TOTAL - iters) / self.cfg.NITER_TOTAL) # big to small
        #     if self.cfg.MULTIPROCESSING_DISTRIBUTED:
        #         content_loss = self.result['loss_content'] * self.cfg.ALPHA_CONTENT * decay_coef
        #         dist.all_reduce(content_loss)
        #     else:
        #         content_loss = self.result['loss_content'].mean() * self.cfg.ALPHA_CONTENT * decay_coef
        #     loss_total += content_loss
        #     content_loss = round(content_loss.item(), 4)
        #     self.loss_meters['TRAIN_SEMANTIC_LOSS'].update(content_loss)

        # if 'PIX2PIX' in self.cfg.LOSS_TYPES:
        #     decay_coef = 1
        #     pix2pix_loss = self.result['loss_pix2pix'].mean() * self.cfg.ALPHA_PIX2PIX * decay_coef
        #     loss_total += pix2pix_loss
        #
        #     pix2pix_loss = round(pix2pix_loss.item(), 4)
        #     self.loss_meters['TRAIN_PIX2PIX_LOSS'].update(pix2pix_loss)

        return loss_total

    def set_log_data(self, cfg):

        self.loss_meters = defaultdict()
        self.log_keys = [
            'TRAIN_G_LOSS',
            'TRAIN_SEMANTIC_LOSS',  # semantic
            'TRAIN_PIX2PIX_LOSS',
            'TRAIN_UNLABELED_LOSS',#converise
            'TRAIN_CLS_ACC',
            'VAL_CLS_ACC',  # classification
            'TRAIN_CLS_LOSS',
            'TRAIN_CLS_MEAN_IOU',
            'VAL_CLS_LOSS',
            'VAL_CLS_MEAN_IOU',
            'VAL_CLS_MEAN_ACC',
            'INTERSECTION',
            'UNION',
            'LABEL'
        ]
        for item in self.log_keys:
            self.loss_meters[item] = AverageMeter()

    def set_optimizer(self, cfg):

        self.optimizers = []

        # if 'PSP' in cfg.MODEL:
        #     self.optimizer = torch.optim.Adam(self.params_list, lr=cfg.LR, betas=(0.5, 0.999))
        # else:
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.LR, betas=(0.5, 0.999))
        # self.optimizer = torch.optim.SGD(self.params_list, lr=cfg.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

        if cfg.MULTIPROCESSING_DISTRIBUTED:
            if cfg.USE_APEX:
                self.net, self.optimizer = apex.amp.initialize(self.net.cuda(), self.optimizer, opt_level=cfg.opt_level)
                self.net = apex.parallel.DistributedDataParallel(self.net)
            else:
                self.net = torch.nn.parallel.DistributedDataParallel(self.net.cuda(), device_ids=[cfg.gpu])

        self.optimizers.append(self.optimizer)

    def validate_seg(self):

        self.phase = 'test'

        # switch to evaluate mode
        self.net.eval()

        intersection_meter = self.loss_meters['INTERSECTION']
        union_meter = self.loss_meters['UNION']
        target_meter = self.loss_meters['LABEL']

        # batch = tqdm(self.val_loader, total=self.val_image_num // self.batch_size_val)
        print('validing model...')
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                self.set_input(data)
                self._forward(cal_loss=False)
                self.pred = self.result['cls'].data.max(1)[1].cpu().numpy()
                label = np.uint8(self._label)

                intersection, union, label = util.intersectionAndUnion(self.pred, label,
                                                                       self.cfg.NUM_CLASSES)
                if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                    dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(label)
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(label)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        self.loss_meters['VAL_CLS_ACC'].update(allAcc)
        self.loss_meters['VAL_CLS_MEAN_ACC'].update(mAcc)
        self.loss_meters['VAL_CLS_MEAN_IOU'].update(mIoU)

    def validate_seg_slide_window(self):

        self.net.eval()
        self.phase = 'test'

        intersection_meter = self.loss_meters['INTERSECTION']
        union_meter = self.loss_meters['UNION']
        target_meter = self.loss_meters['LABEL']

        print('testing with sliding windows...')
        num_images = 0

        # batch = tqdm(self.val_loader, total=self.val_image_num // self.batch_size_val)
        # for data in batch:
        for data in self.val_loader:
            self.set_input(data)
            num_images += self.batch_size
            pred = util.slide_cal(model=self.net, image=self.source_modal, crop_size=self.cfg.FINE_SIZE,
                                  prediction_matrix=self.prediction_matrix[0:self.batch_size, :, :, :],
                                  count_crop_matrix=self.count_crop_matrix[0:self.batch_size, :, :, :])

            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                self.pred = pred.data.max(1)[1]
                intersection, union, label = util.intersectionAndUnionGPU(self.pred, self.label, self.cfg.NUM_CLASSES)
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(label)
                intersection, union, label = intersection.cpu().numpy(), union.cpu().numpy(), label.cpu().numpy()
            else:
                self.pred = pred.data.max(1)[1].cpu().numpy()
                # self.pred = np.argmax(pred, axis=1)
                intersection, union, label = util.intersectionAndUnion(self.pred, np.uint8(self._label),
                                                                       self.cfg.NUM_CLASSES)

            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(label)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        self.loss_meters['VAL_CLS_ACC'].update(allAcc)
        self.loss_meters['VAL_CLS_MEAN_ACC'].update(mAcc)
        self.loss_meters['VAL_CLS_MEAN_IOU'].update(mIoU)
        print('test images num: ', num_images)

    def write_loss(self, phase, global_step):


        if self.cfg.MODEL=='contrastive':
            self.writer.add_image('Contrastive/image',
                              torchvision.utils.make_grid(self.source_modal[:1].clone().cpu().data, 3,
                                                        normalize=True), global_step=global_step)
            self.writer.add_scalar('Contrastive_loss/LR', self.optimizer.param_groups[0]['lr'], global_step=global_step)
            self.writer.add_scalar('Contrastive_loss/loss', self.loss_meters['TRAIN_UNLABELED_LOSS'].avg, global_step=global_step)

            if isinstance(self.result['l_gen'], list):
                for i, (gen_l,gen_ab, lab_label) in enumerate(zip(self.result['l_gen'],self.result['ab_gen'],self.target_modal)):
                    b,c,h,w = gen_l.size()
                    pic = np.zeros((1,3,h,w))
                    l_label,ab_label=torch.split(lab_label,[1,2],dim=1)
                    # pic[:,2:,:,:] += 255
                    self.writer.add_image('Contrastive/Gen_l' + str(self.cfg.FINE_SIZE / pow(2, i)),
                                          torchvision.utils.make_grid(gen_l[:1].clone().cpu().data, 1,normalize=True),
                                          global_step=global_step)
                    pic[:,1:,:,:] = gen_ab[:1].clone().cpu().data
                    self.writer.add_image('Contrastive/Gen_ab' + str(self.cfg.FINE_SIZE / pow(2, i)),
                                          torchvision.utils.make_grid(torch.from_numpy(pic), 3,normalize=True),
                                          global_step=global_step)
                    self.writer.add_image('Contrastive/target_l' + str(self.cfg.FINE_SIZE / pow(2, i)),
                                          torchvision.utils.make_grid(l_label[:1].clone().cpu().data, 1,normalize=True),
                                          global_step=global_step)
                    pic[:,1:,:,:] = ab_label[:1].clone().cpu().data
                    self.writer.add_image('Contrastive/target_ab' + str(self.cfg.FINE_SIZE / pow(2, i)),
                                          torchvision.utils.make_grid(torch.from_numpy(pic), 3,normalize=True),
                                          global_step=global_step)
            return


        loss_types = self.cfg.LOSS_TYPES
        if self.phase == 'train':
            label_show = self.label.data.cpu().numpy()
        else:
            label_show = np.uint8(self.label.data.cpu())

        source_modal_show = self.source_modal
        target_modal_show = self.target_modal
        train_pred = self.result['cls'].data.max(1)[1].cpu().numpy()

        if phase == 'train':
            self.writer.add_image('Seg/Train_image',
                                  torchvision.utils.make_grid(source_modal_show[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)
            self.writer.add_scalar('Seg/LR', self.optimizer.param_groups[0]['lr'], global_step=global_step)

            if 'CLS' in loss_types:
                self.writer.add_scalar('Seg/TRAIN_CLS_LOSS', self.loss_meters['TRAIN_CLS_LOSS'].avg,
                                       global_step=global_step)
                # self.writer.add_scalar('TRAIN_CLS_ACC', self.loss_meters['TRAIN_CLS_ACC'].avg*100.0,
                #                        global_step=global_step)
                # self.writer.add_scalar('TRAIN_CLS_MEAN_IOU', float(self.train_iou.mean())*100.0,
                #                        global_step=global_step)

            if self.trans and not self.cfg.MULTI_MODAL:

                if 'SEMANTIC' in self.cfg.LOSS_TYPES:
                    self.writer.add_scalar('Seg/TRAIN_SEMANTIC_LOSS', self.loss_meters['TRAIN_SEMANTIC_LOSS'].avg,
                                           global_step=global_step)
                if 'PIX2PIX' in self.cfg.LOSS_TYPES:
                    self.writer.add_scalar('Seg/TRAIN_PIX2PIX_LOSS', self.loss_meters['TRAIN_PIX2PIX_LOSS'].avg,
                                           global_step=global_step)

                self.writer.add_image('Seg/Train_image',
                                      torchvision.utils.make_grid(source_modal_show[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)
                # if isinstance(self.target_modal, list):
                #     for i, (gen, target) in enumerate(zip(self.gen, self.target_modal)):
                #         self.writer.add_image('Seg/2_Train_Gen_' + str(self.cfg.FINE_SIZE / pow(2, i)),
                #                               torchvision.utils.make_grid(gen[:6].clone().cpu().data, 3,
                #                                                           normalize=True),
                #                               global_step=global_step)
                #         self.writer.add_image('Seg/3_Train_Target_' + str(self.cfg.FINE_SIZE / pow(2, i)),
                #                               torchvision.utils.make_grid(target[:6].clone().cpu().data, 3,
                #                                                           normalize=True),
                #                               global_step=global_step)
                # else:
                self.writer.add_image('Seg/Train_target',
                                      torchvision.utils.make_grid(target_modal_show[:6].clone().cpu().data, 3,
                
                
                                                                  normalize=True), global_step=global_step)
                self.writer.add_image('Seg/Train_gen',
                                      torchvision.utils.make_grid(self.gen.data[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)

            if 'CLS' in loss_types:
                self.writer.add_image('Seg/Train_predicted',
                                      torchvision.utils.make_grid(
                                          torch.from_numpy(util.color_label(train_pred[:6],
                                                                            ignore=self.cfg.IGNORE_LABEL,
                                                                            dataset=self.cfg.DATASET)), 3,
                                          normalize=True, range=(0, 255)), global_step=global_step)
                self.writer.add_image('Seg/Train_label',
                                      torchvision.utils.make_grid(
                                          torch.from_numpy(
                                              util.color_label(label_show[:6], ignore=self.cfg.IGNORE_LABEL,
                                                               dataset=self.cfg.DATASET)), 3, normalize=True,
                                          range=(0, 255)), global_step=global_step)

        if phase == 'test':
            # self.writer.add_image('Seg/Val_image',
            #                       torchvision.utils.make_grid(source_modal_show[:6].clone().cpu().data, 3,
            #                                                   normalize=True), global_step=global_step)
            #
            # self.writer.add_image('Seg/Val_predicted',
            #                       torchvision.utils.make_grid(
            #                           torch.from_numpy(util.color_label(self.pred[:6], ignore=self.cfg.IGNORE_LABEL,
            #                                                             dataset=self.cfg.DATASET)), 3,
            #                           normalize=True, range=(0, 255)), global_step=global_step)
            # self.writer.add_image('Seg/Val_label',
            #                       torchvision.utils.make_grid(torch.from_numpy(
            #                           util.color_label(label_show[:6], ignore=self.cfg.IGNORE_LABEL,
            #                                            dataset=self.cfg.DATASET)),
            #                           3, normalize=True, range=(0, 255)),
            #                       global_step=global_step)

            self.writer.add_scalar('Seg/VAL_CLS_ACC', self.loss_meters['VAL_CLS_ACC'].val * 100.0,
                                   global_step=global_step)
            self.writer.add_scalar('Seg/VAL_CLS_MEAN_ACC', self.loss_meters['VAL_CLS_MEAN_ACC'].val * 100.0,
                                   global_step=global_step)
            self.writer.add_scalar('Seg/VAL_CLS_MEAN_IOU', self.loss_meters['VAL_CLS_MEAN_IOU'].val * 100.0,
                                   global_step=global_step)
