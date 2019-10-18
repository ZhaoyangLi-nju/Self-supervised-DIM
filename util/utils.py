import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from collections import OrderedDict


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_images(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)
    image_names = [d for d in os.listdir(dir)]
    for image_name in image_names:
        if has_file_allowed_extension(image_name, extensions):
            file = os.path.join(dir, image_name)
            images.append(file)
    return images


# Checks if a file is an allowed extension.
def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def mean_acc(target_indice, pred_indice, num_classes, classes=None):
    assert (num_classes == len(classes))
    acc = 0.
    print('{0} Class Acc Report {1}'.format('#' * 10, '#' * 10))
    for i in range(num_classes):
        idx = np.where(target_indice == i)[0]
        # acc = acc + accuracy_score(target_indice[idx], pred_indice[idx])
        class_correct = accuracy_score(target_indice[idx], pred_indice[idx])
        acc += class_correct
        print('acc {0}: {1:.3f}'.format(classes[i], class_correct * 100))

        # class report
        # y_tpye, y_true, y_pred = _check_targets(target_indice[idx], pred_indice[idx])
        # score = y_true == y_pred
        # wrong_index = np.where(score == False)[0]
        # for j in idx[wrong_index]:
        #     print("Wrong for class [%s]: predicted as: <%s>, image_id--<%s>" %
        #           (int_to_class[i], int_to_class[pred[j]], image_paths[j]))
        #
        # print("[class] %s accuracy is %.3f" % (int_to_class[i], class_correct))
    print('#' * 30)
    return (acc / num_classes) * 100


def process_output(output):
    # Computes the result and argmax index
    pred, index = output.topk(1, 1, largest=True)

    return pred.cpu().float().numpy().flatten(), index.cpu().numpy().flatten()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        if weight:
            weight = torch.FloatTensor(weight).cuda()
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)


def accuracy(preds, label):
    valid = (label > 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / float(valid_sum + 1e-10)
    return acc, valid_sum


def color_label_np(label, ignore=None, dataset=None):
    if dataset == 'cityscapes':
        label_colours = label_colours_cityscapes
    elif dataset == 'sunrgbd':
        label_colours = label_colours_sunrgbd
    colored_label = np.vectorize(lambda x: label_colours[-1] if x == ignore else label_colours[int(x)])
    colored = np.asarray(colored_label(label)).astype(np.float32)
    # colored = colored.squeeze()

    try:
        return colored.transpose([1, 2, 0])
    except ValueError:
        return colored[np.newaxis, ...]


def color_label(label, ignore=None, dataset=None):
    # label = label.data.cpu().numpy()
    if dataset == 'cityscapes':
        label_colours = label_colours_cityscapes
    elif dataset == 'sunrgbd':
        label_colours = label_colours_sunrgbd
    colored_label = np.vectorize(lambda x: label_colours[-1] if x == ignore else label_colours[int(x)])
    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    try:
        return colored.transpose([1, 0, 2, 3])
    except ValueError:
        return colored[np.newaxis, ...]


def get_current_errors(meter_dics, current=True):
    loss_dict = OrderedDict()
    for key, value in sorted(meter_dics.items(), reverse=True):

        if 'TEST' in key or 'VAL' in key or 'ACC' in key or value.val == 0 or 'LAYER' in key:
            continue
        if current:
            loss_dict[key] = value.val
        else:
            loss_dict[key] = value.avg
    return loss_dict


def print_current_errors(errors, epoch, i=None, t=None):
    print('#' * 10)
    if i is None:
        message = '(Training Loss_avg [Epoch:{0}]) '.format(epoch)
    else:
        message = '(epoch: {epoch}, iters: {iter}, time: {time:.3f}) '.format(epoch=epoch, iter=i, time=t)

    for k, v in errors.items():
        if 'CLS' in k and i is None:
            message += '{key}: [{value:.3f}] '.format(key=k, value=v)
        else:
            message += '{key}: {value:.3f} '.format(key=k, value=v)
    print(message)
    print('#' * 10)


def evaluate(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def intersectionAndUnion(output, label, num_classes, ignore_index=255):
    # 'K' classes, output and label sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    output = output.reshape(output.size)
    label = label.reshape(label.size)
    output[np.where(label == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == label)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(num_classes + 1))
    area_output, _ = np.histogram(output, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_output + area_label - area_intersection
    return area_intersection, area_union, area_label

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()

def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def slide_cal(model, image, crop_size, stride_rate=2 / 3, prediction_matrix=None, count_crop_matrix=None):
    crop_h, crop_w = crop_size
    batch_size, _, h, w = image.size()
    assert crop_h <= h
    assert crop_w <= w
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(w - crop_w) / stride_w) + 1)

    prediction_matrix.fill_(0)
    count_crop_matrix.fill_(0)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, w)
            s_w = e_w - crop_w
            image_crop = image[:, :, s_h:e_h, s_w:e_w]
            count_crop_matrix[:, :, s_h:e_h, s_w:e_w] += 1
            with torch.no_grad():
                result = model(source=image_crop, phase='test', cal_loss=False)
                prediction_matrix[:, :, s_h:e_h, s_w:e_w] += result['cls']

    prediction_matrix /= count_crop_matrix
    return prediction_matrix


def get_confusion_matrix(gt_label, pred_label, class_num=37):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


label_colours_sunrgbd = [
    (148, 65, 137), (255, 116, 69), (86, 156, 137),
    (202, 179, 158), (155, 99, 235), (161, 107, 108),
    (133, 160, 103), (76, 152, 126), (84, 62, 35),
    (44, 80, 130), (31, 184, 157), (101, 144, 77),
    (23, 197, 62), (141, 168, 145), (142, 151, 136),
    (115, 201, 77), (100, 216, 255), (57, 156, 36),
    (88, 108, 129), (105, 129, 112), (42, 137, 126),
    (155, 108, 249), (166, 148, 143), (81, 91, 87),
    (100, 124, 51), (73, 131, 121), (157, 210, 220),
    (134, 181, 60), (221, 223, 147), (123, 108, 131),
    (161, 66, 179), (163, 221, 160), (31, 146, 98),
    (99, 121, 30), (49, 89, 240), (116, 108, 9),
    (139, 110, 246), (0, 0, 0)
]  # list(-1) for 255

label_colours_cityscapes = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 0, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
    (0, 0, 0)
]
