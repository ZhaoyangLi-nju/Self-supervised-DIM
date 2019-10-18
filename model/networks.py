import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .resnet import resnet18
from .resnet import resnet50
from .resnet import resnet101
# import torchvision.models as models
import model.resnet as models

# batch_norm = nn.BatchNorm2d

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def fix_grad(net):
    print(net.__class__.__name__)

    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm2d') != -1:
            m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False

    net.apply(fix_func)


def unfix_grad(net):
    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm2d') != -1 or classname.find('Linear') != -1:
            m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True

    net.apply(fix_func)


def define_netowrks(cfg, device=None,Batch_norm=None):
    if 'resnet' in cfg.ARCH:

        # sync bn or not
        # models.BatchNorm = batch_norm

        if cfg.MULTI_SCALE:
            if cfg.MODEL == 'contrastive':
                model = Contrastive_CrossModal_Conc(cfg,device=device)
            # model = FCN_Conc_Multiscale(cfg, device=device)
            # pass
        elif cfg.MULTI_MODAL:
            # model = FCN_Conc_MultiModalTarget_Conc(cfg, device=device)
            model = FCN_Conc_MultiModalTarget(cfg, device=device)
            # model = FCN_Conc_MultiModalTarget_Late(cfg, device=device)
        else:
            if cfg.MODEL == 'FCN':
                model = FCN_Conc(cfg, device=device)
            if cfg.MODEL == 'FCN_MAXPOOL':
                model = FCN_Conc_Maxpool(cfg, device=device)
            # if cfg.MODEL == 'FCN_LAT':
            #     model = FCN_Conc_Lat(cfg, device=device)
            elif cfg.MODEL == 'UNET':
                model = UNet(cfg, device=device)
            # elif cfg.MODEL == 'UNET_256':
            #     model = UNet_Share_256(cfg, device=device)
            # elif cfg.MODEL == 'UNET_128':
            #     model = UNet_Share_128(cfg, device=device)
            # elif cfg.MODEL == 'UNET_64':
            #     model = UNet_Share_64(cfg, device=device)
            # elif cfg.MODEL == 'UNET_LONG':
            #     model = UNet_Long(cfg, device=device)
            elif cfg.MODEL == "PSP":
                model = PSPNet(cfg, Batch_norm,device=device)

    return model


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv_norm_relu(dim_in, dim_out, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d,
                   use_leakyRelu=False, use_bias=False, is_Sequential=True):
    if use_leakyRelu:
        act = nn.LeakyReLU(0.2, True)
    else:
        act = nn.ReLU(True)

    if is_Sequential:
        result = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=use_bias),
            norm(dim_out, affine=True),
            act
        )
        return result
    return [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            norm(dim_out, affine=True),
            act]


def expand_Conv(module, in_channels):
    def expand_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.in_channels = in_channels
            m.out_channels = m.out_channels
            mean_weight = torch.mean(m.weight, dim=1, keepdim=True)
            m.weight.data = mean_weight.repeat(1, in_channels, 1, 1).data

    module.apply(expand_func)


##############################################################################
# Moduels
##############################################################################
# class Upsample_Interpolate(nn.Module):
#
#     def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, norm=nn.BatchNorm2d, scale=2, mode='bilinear'):
#         super(Upsample_Interpolate, self).__init__()
#         self.scale = scale
#         self.mode = mode
#         self.conv_norm_relu1 = conv_norm_relu(dim_in, dim_out, kernel_size=kernel_size, stride=1, padding=padding,
#                                               norm=norm)
#         self.conv_norm_relu2 = conv_norm_relu(dim_out, dim_out, kernel_size=3, stride=1, padding=1, norm=norm)
#
#     def forward(self, x, activate=True):
#         x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
#         x = self.conv_norm_relu1(x)
#         x = self.conv_norm_relu2(x)
#         return x


# class UpConv_Conc(nn.Module):
#
#     def __init__(self, dim_in, dim_out, scale=2, mode='bilinear', norm=nn.BatchNorm2d, if_conc=True):
#         super(UpConv_Conc, self).__init__()
#         self.scale = scale
#         self.mode = mode
#         self.up = nn.Sequential(
#             nn.UpsamplingBilinear2d(scale_factor=scale),
#             conv_norm_relu(dim_in, dim_out, kernel_size=1, padding=0, norm=norm)
#             # nn.Conv2d(dim_in, dim_out, 1, bias=False),
#         )
#         if if_conc:
#             dim_in = dim_out * 2
#         self.conc = nn.Sequential(
#             conv_norm_relu(dim_in, dim_out, kernel_size=1, padding=0, norm=norm),
#             conv_norm_relu(dim_out, dim_out, kernel_size=3, padding=1, norm=norm)
#         )
#
#     def forward(self, x, y=None):
#         x = self.up(x)
#         residual = x
#         if y is not None:
#             x = torch.cat((x, y), 1)
#         return self.conc(x) + residual


# class UpsampleBasicBlock(nn.Module):
#
#     def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d, scale=2, mode='bilinear', upsample=True):
#         super(UpsampleBasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
#                       padding=padding, bias=False)
#         self.bn1 = norm(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm(planes)
#
#         if inplanes != planes:
#             kernel_size, padding = 1, 0
#         else:
#             kernel_size, padding = 3, 1
#
#         if upsample:
#
#             self.trans = nn.Sequential(
#                 nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
#                           padding=padding, bias=False),
#                 norm(planes))
#         else:
#             self.trans = None
#
#         self.scale = scale
#         self.mode = mode
#
#     def forward(self, x):
#
#         if self.trans is not None:
#             x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
#             residual = self.trans(x)
#         else:
#             residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += residual
#
#         return out

class Conc_Up_Residual(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d, conc_feat=True):
        super(Conc_Up_Residual, self).__init__()

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1,
                      padding=0, bias=False),
            norm(dim_out))
        self.smooth = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,
                                padding=1, bias=False)

        if conc_feat:
            dim_in = dim_out * 2
            kernel_size, padding = 1, 0
        else:
            kernel_size, padding = 3, 1

        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.norm1 = norm(dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_out, dim_out)
        self.norm2 = norm(dim_out)

    def forward(self, x, y=None):

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.smooth(x)
        residual = self.residual_conv(x)

        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)

        x += residual

        return self.relu(x)


class Conc_Up_Residual_bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d, conc_feat=True):
        super(Conc_Up_Residual_bottleneck, self).__init__()

        self.smooth = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1,
                                padding=0, bias=False)

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            norm(dim_out))

        if conc_feat:
            dim_in = dim_out * 2
        else:
            dim_in = dim_out

        dim_med = int(dim_out / 2)
        self.conv1 = nn.Conv2d(dim_in, dim_med, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(dim_out)

    def forward(self, x, y=None):

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.smooth(x)
        residual = self.residual_conv(x)

        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        x += residual

        return self.relu(x)

class Conc_Residual_bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d, conc_feat=True):
        super(Conc_Residual_bottleneck, self).__init__()

        self.conv0 = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            norm(dim_out))
        # else:
        #     self.residual_conv = nn.Sequential(
        #         nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=2,
        #                   padding=1, bias=False),
        #         norm(dim_out))

        if conc_feat:
            dim_in = dim_out * 2
        dim_med = int(dim_out / 2)
        self.conv1 = nn.Conv2d(dim_in, dim_med, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(dim_out)

    def forward(self, x, y=None):

        x = self.conv0(x)
        residual = self.residual_conv(x)

        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        x += residual

        return self.relu(x)


class Lat_Up_Residual(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d):
        super(Lat_Up_Residual, self).__init__()

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1,
                      padding=0, bias=False),
            norm(dim_out))
        self.smooth = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,
                                padding=1, bias=False)

        self.lat = nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1,
                             padding=0, bias=False)

        self.conv1 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.norm1 = norm(dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.norm2 = norm(dim_out)

    def forward(self, x, y=None):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.smooth(x)
        residual = self.residual_conv(x)

        if y is not None:
            x = x + self.lat(y)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)

        x += residual

        return self.relu(x)


#########################################

##############################################################################
# Translate to recognize
##############################################################################
class Content_Model(nn.Module):

    def __init__(self, cfg, criterion=None, in_channel=3):
        super(Content_Model, self).__init__()
        self.cfg = cfg
        self.criterion = criterion
        self.net = cfg.WHICH_CONTENT_NET

        if 'resnet' in self.net:
            from .pretrained_resnet import ResNet
            self.model = ResNet(self.net, cfg, in_channel=in_channel)

        fix_grad(self.model)
        # print_network(self)

    def forward(self, x, target, layers=None):

        # important when set content_model as the attr of trecg_net
        self.model.eval()

        layers = layers
        if layers is None or not layers:
            layers = self.cfg.CONTENT_LAYERS.split(',')

        input_features = self.model((x + 1) / 2, layers)
        target_targets = self.model((target + 1) / 2, layers)
        len_layers = len(layers)
        loss_fns = [self.criterion] * len_layers
        alpha = [1] * len_layers

        content_losses = [alpha[i] * loss_fns[i](gen_content, target_targets[i])
                          for i, gen_content in enumerate(input_features)]
        loss = sum(content_losses)
        return loss


# class FCN_Lat(nn.Module):
#
#     def __init__(self, cfg, device=None):
#         super(FCN_Lat, self).__init__()
#
#         self.cfg = cfg
#         self.trans = not cfg.NO_TRANS
#         self.device = device
#         encoder = cfg.ARCH
#         num_classes = cfg.NUM_CLASSES
#
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
#
#         if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
#             pretrained = True
#         else:
#             pretrained = False
#
#         if cfg.PRETRAINED == 'place':
#             resnet = models.__dict__['resnet18'](num_classes=365)
#             load_path = "./initmodel/resnet18_places365.pth"
#             checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
#             state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
#             resnet.load_state_dict(state_dict)
#             print('place resnet18 loaded....')
#         else:
#             resnet = resnet18(pretrained=pretrained)
#             print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
#
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool  # 1/4
#         self.layer1 = resnet.layer1  # 1/4
#         self.layer2 = resnet.layer2  # 1/8
#         self.layer3 = resnet.layer3  # 1/16
#         self.layer4 = resnet.layer4  # 1/32
#         self.head = _FCNHead(512, num_classes, nn.BatchNorm2d)
#         # self.head = nn.Conv2d(512, num_classes, 1)
#
#         if self.using_semantic_branch:
#             self.build_upsample_content_layers(dims, num_classes)
#
#         self.score_aux1 = nn.Conv2d(256, num_classes, 1)
#         self.score_aux2 = nn.Conv2d(128, num_classes, 1)
#         self.score_aux3 = nn.Conv2d(64, num_classes, 1)
#
#         # self.avgpool = nn.AvgPool2d(self.avg_pool_size, 1)
#         # self.fc = nn.Linear(fc_input_nc, cfg.NUM_CLASSES)
#
#         if pretrained:
#             init_weights(self.head, 'normal')
#
#             if self.trans:
#                 init_weights(self.lat1, 'normal')
#                 init_weights(self.lat2, 'normal')
#                 init_weights(self.lat3, 'normal')
#                 init_weights(self.up1, 'normal')
#                 init_weights(self.up2, 'normal')
#                 init_weights(self.up3, 'normal')
#                 init_weights(self.up4, 'normal')
#
#             init_weights(self.head, 'normal')
#             init_weights(self.score_aux3, 'normal')
#             init_weights(self.score_aux2, 'normal')
#             init_weights(self.score_aux1, 'normal')
#
#         else:
#
#             init_weights(self, 'normal')
#
#     def set_content_model(self, content_model):
#         self.content_model = content_model
#
#     def set_pix2pix_criterion(self, criterion):
#         self.pix2pix_criterion = criterion.to(self.device)
#
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
#
#     def build_upsample_content_layers(self, dims, num_classes):
#
#         norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
#
#         self.up1 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm, conc_feat=False)
#         self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm, conc_feat=False)
#         self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)
#         self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
#
#         self.lat1 = nn.Conv2d(dims[3], dims[3], kernel_size=1, stride=1, padding=0, bias=False)
#         self.lat2 = nn.Conv2d(dims[2], dims[2], kernel_size=1, stride=1, padding=0, bias=False)
#         self.lat3 = nn.Conv2d(dims[1], dims[1], kernel_size=1, stride=1, padding=0, bias=False)
#
#         self.up_image_content = nn.Sequential(
#             nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
#
#         self.score_up_256 = nn.Sequential(
#             nn.Conv2d(256, num_classes, 1)
#         )
#
#         self.score_up_128 = nn.Sequential(
#             nn.Conv2d(128, num_classes, 1)
#         )
#         self.score_up_64 = nn.Sequential(
#             nn.Conv2d(64, num_classes, 1)
#         )
#
#     def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
#                 return_losses=True):
#         result = {}
#
#         layer_0 = self.relu(self.bn1(self.conv1(source)))
#         if not self.trans:
#             layer_0 = self.maxpool(layer_0)
#         layer_1 = self.layer1(layer_0)
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
#
#         if self.trans:
#             # content model branch
#             skip_1 = self.lat1(layer_3)
#             skip_2 = self.lat2(layer_2)
#             skip_3 = self.lat3(layer_1)
#
#             up1 = self.up1(layer_4)
#             up2 = self.up2(up1 + skip_1)
#             up3 = self.up3(up2 + skip_2)
#             up4 = self.up4(up3 + skip_3)
#
#             result['gen_img'] = self.up_image_content(up4)
#             if phase == 'train':
#                 result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)
#
#         if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.INFERENCE:
#             # segmentation branch
#             score_head = self.head(layer_4)
#
#             if self.cfg.WHICH_SCORE == 'main' or not self.trans:
#                 score_aux1 = self.score_aux1(layer_3)
#                 score_aux2 = self.score_aux2(layer_2)
#                 score_aux3 = self.score_aux3(layer_1)
#             elif self.cfg.WHICH_SCORE == 'up':
#                 score_aux1 = self.score_aux1(up1)
#                 score_aux2 = self.score_aux2(up2)
#                 score_aux3 = self.score_aux3(up3)
#             elif self.cfg.WHICH_SCORE == 'both':
#                 score_aux1 = self.score_aux1(up1 + layer_3)
#                 score_aux2 = self.score_aux2(up2 + layer_2)
#                 score_aux3 = self.score_aux3(up3 + layer_1)
#
#             score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
#             score = score + score_aux1
#             score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
#             score = score + score_aux2
#             score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
#             score = score + score_aux3
#
#             result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
#
#             if phase == 'train':
#                 result['loss_cls'] = self.cls_criterion(result['cls'], label)
#
#         return result


class FCN_Conc(nn.Module):

    def __init__(self, cfg, device=None):
        super(FCN_Conc, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.resnet18(num_classes=365)
            load_path = "./initmodel/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = models.__dict__[cfg.ARCH](pretrained=pretrained, deep_base=False)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))


        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

        if self.trans:
            self.build_upsample_content_layers(dims)

        if 'resnet18' == cfg.ARCH:
            head_dim = 512
            aux_dims = [256, 128, 64]
        elif 'resnet50' == cfg.ARCH:
            head_dim = 2048
            aux_dims = [1024, 512, 256]

        self.head = _FCNHead(head_dim, num_classes, nn.BatchNorm2d)

        self.score_aux1 = nn.Sequential(
            nn.Conv2d(aux_dims[0], num_classes, 1)
        )

        self.score_aux2 = nn.Sequential(
            nn.Conv2d(aux_dims[1], num_classes, 1)
        )
        self.score_aux3 = nn.Sequential(
            nn.Conv2d(aux_dims[2], num_classes, 1)
        )

        if pretrained:
            init_weights(self.head, 'normal')

            if self.trans:
                init_weights(self.up1, 'normal')
                init_weights(self.up2, 'normal')
                init_weights(self.up3, 'normal')
                init_weights(self.up4, 'normal')

            init_weights(self.head, 'normal')
            init_weights(self.score_aux3, 'normal')
            init_weights(self.score_aux2, 'normal')
            init_weights(self.score_aux1, 'normal')

        else:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
        
        if 'resnet18' == self.cfg.ARCH:
            self.up1 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
            self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
            self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
    
        elif 'resnet50' in self.cfg.ARCH:
            self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
            self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
            self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[1], norm=norm, conc_feat=False)

        self.up_image_content = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
        result = {}

        layer_0 = self.relu(self.bn1(self.conv1(source)))
        if not self.trans:
            layer_0 = self.maxpool(layer_0)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        if self.trans:
            # translation branch

            up1 = self.up1(layer_4, layer_3)
            up2 = self.up2(up1, layer_2)
            up3 = self.up3(up2, layer_1)
            up4 = self.up4(up3)

            result['gen_img'] = self.up_image_content(up4)

            if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

        if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.INFERENCE:

            # segmentation branch
            score_head = self.head(layer_4)

            score_aux1 = None
            score_aux2 = None
            score_aux3 = None
            if self.cfg.WHICH_SCORE == 'main' or not self.trans:
                score_aux1 = self.score_aux1(layer_3)
                score_aux2 = self.score_aux2(layer_2)
                score_aux3 = self.score_aux3(layer_1)
            elif self.cfg.WHICH_SCORE == 'up':
                score_aux1 = self.score_aux1(up1)
                score_aux2 = self.score_aux2(up2)
                score_aux3 = self.score_aux3(up3)

            score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux1
            score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux2
            score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux3

            result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            if cal_loss:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


# class FCN_Conc_Resnet50(nn.Module):
# 
#     def __init__(self, cfg, device=None):
#         super(FCN_Conc_Resnet50, self).__init__()
# 
#         self.cfg = cfg
#         self.trans = not cfg.NO_TRANS
#         self.device = device
#         encoder = cfg.ARCH
#         num_classes = cfg.NUM_CLASSES
# 
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
# 
#         if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
#             pretrained = True
#         else:
#             pretrained = False
# 
#         # models.BatchNorm = SyncBatchNorm
#         resnet = models.__dict__[cfg.ARCH](pretrained=pretrained, deep_base=False)
#         print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
# 
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool  # 1/4
#         self.layer1 = resnet.layer1  # 1/4
#         self.layer2 = resnet.layer2  # 1/8
#         self.layer3 = resnet.layer3  # 1/16
#         self.layer4 = resnet.layer4  # 1/32
#         self.head = _FCNHead(2048, num_classes, nn.BatchNorm2d)
#         # self.head = nn.Conv2d(512, num_classes, 1)
# 
#         if self.trans:
#             self.build_upsample_content_layers(dims)
# 
#         self.score_1024 = nn.Sequential(
#             nn.Conv2d(dims[5], num_classes, 1)
#         )
#         self.score_head = nn.Sequential(
#             nn.Conv2d(dims[4], num_classes, 1)
#         )
#         self.score_aux1 = nn.Sequential(
#             nn.Conv2d(dims[3], num_classes, 1)
#         )
# 
#         if pretrained:
#             init_weights(self.head, 'normal')
# 
#             if self.trans:
#                 init_weights(self.up1, 'normal')
#                 init_weights(self.up2, 'normal')
#                 init_weights(self.up3, 'normal')
#                 init_weights(self.up4, 'normal')
# 
#             init_weights(self.head, 'normal')
#             init_weights(self.score_1024, 'normal')
#             init_weights(self.score_aux1, 'normal')
#             init_weights(self.score_head, 'normal')
# 
#         else:
# 
#             init_weights(self, 'normal')
# 
#     def set_content_model(self, content_model):
#         self.content_model = content_model
# 
#     def set_pix2pix_criterion(self, criterion):
#         self.pix2pix_criterion = criterion.to(self.device)
# 
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
# 
#     def build_upsample_content_layers(self, dims):
# 
#         norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
# 
#         self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
#         self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
#         self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
#         self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[1], norm=norm, conc_feat=False)
# 
#         self.up_image_content = nn.Sequential(
#             nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
# 
#     def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
#                 return_losses=True):
#         result = {}
# 
#         layer_0 = self.relu(self.bn1(self.conv1(source)))
#         if not self.trans:
#             layer_0 = self.maxpool(layer_0)
#         layer_1 = self.layer1(layer_0)
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
# 
#         if self.trans:
#             # content model branch
# 
#             # up1 = self.up1(layer_4)
#             up1 = self.up1(layer_4, layer_3)
#             up2 = self.up2(up1, layer_2)
#             up3 = self.up3(up2, layer_1)
#             up4 = self.up4(up3)
# 
#             result['gen_img'] = self.up_image_content(up4)
# 
#         # segmentation branch
#         # score_2048 = nn.Conv2d(2048, self.cfg.NUM_CLASSES, 1)
#         score_2048 = self.head(layer_4)
# 
#         score_1024 = None
#         score_head = None
#         score_aux1 = None
#         if self.cfg.WHICH_SCORE == 'main' or not self.trans:
#             score_1024 = self.score_1024(layer_3)
#             score_head = self.score_head(layer_2)
#             score_aux1 = self.score_aux1(layer_1)
#         elif self.cfg.WHICH_SCORE == 'up':
#             score_1024 = self.score_1024(up1)
#             score_head = self.score_head(up2)
#             score_aux1 = self.score_aux1(up3)
# 
#         score = F.interpolate(score_2048, score_1024.size()[2:], mode='bilinear', align_corners=True)
#         score = score + score_1024
#         score = F.interpolate(score, score_head.size()[2:], mode='bilinear', align_corners=True)
#         score = score + score_head
#         score = F.interpolate(score, score_aux1.size()[2:], mode='bilinear', align_corners=True)
#         score = score + score_aux1
# 
#         result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
# 
#         if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
#             result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)
# 
#         if 'CLS' in self.cfg.LOSS_TYPES and phase == 'train':
#             result['loss_cls'] = self.cls_criterion(result['cls'], label)
# 
#         if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
#             result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)
# 
#         return result


class FCN_Conc_Maxpool(nn.Module):

    def __init__(self, cfg, device=None):
        super(FCN_Conc_Maxpool, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "./initmodel/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = models.__dict__[cfg.ARCH](pretrained=pretrained, deep_base=False)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        # self.head = nn.Conv2d(512, num_classes, 1)

        if self.trans:
            if 'resnet50' in self.cfg.ARCH:
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            elif 'resnet18' in self.cfg.ARCH:
                for n, m in self.layer4.named_modules():
                    if 'conv1' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            self.build_upsample_content_layers(dims)

        if 'resnet18' == cfg.ARCH:
            aux_dims = [256, 128, 64]
            head_dim = 512
        elif 'resnet50' == cfg.ARCH:
            aux_dims = [1024, 512, 256]
            head_dim = 2048

        self.head = _FCNHead(head_dim, num_classes, nn.BatchNorm2d)

        self.score_aux1 = nn.Sequential(
            nn.Conv2d(aux_dims[0], num_classes, 1)
        )

        self.score_aux2 = nn.Sequential(
            nn.Conv2d(aux_dims[1], num_classes, 1)
        )
        self.score_aux3 = nn.Sequential(
            nn.Conv2d(aux_dims[2], num_classes, 1)
        )

        if pretrained:
            init_weights(self.head, 'normal')

            if self.trans:
                init_weights(self.up1, 'normal')
                init_weights(self.up2, 'normal')
                init_weights(self.up3, 'normal')
                init_weights(self.up4, 'normal')
                init_weights(self.up5, 'normal')

            init_weights(self.head, 'normal')
            init_weights(self.score_aux3, 'normal')
            init_weights(self.score_aux2, 'normal')
            init_weights(self.score_aux1, 'normal')

        else:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        if 'resnet18' == self.cfg.ARCH:
            self.up1 = Conc_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
            self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
            self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm)
            self.up5 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)

        elif 'resnet50' in self.cfg.ARCH:
            self.up1 = Conc_Residual_bottleneck(dims[6], dims[5], norm=norm)
            self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
            self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[1], norm=norm)
            self.up5 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)

        self.up_image_content = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
        result = {}

        layer_0 = self.relu(self.bn1(self.conv1(source)))
        layer_1 = self.layer1(self.maxpool(layer_0))
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        if self.trans:
            # translation branch

            up1 = self.up1(layer_4, layer_3)
            up2 = self.up2(up1, layer_2)
            up3 = self.up3(up2, layer_1)
            up4 = self.up4(up3, layer_0)
            up5 = self.up5(up4)

            result['gen_img'] = self.up_image_content(up5)

            if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

        if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.INFERENCE:

            # segmentation branch
            score_head = self.head(layer_4)

            score_aux1 = None
            score_aux2 = None
            score_aux3 = None
            if self.cfg.WHICH_SCORE == 'main' or not self.trans:
                score_aux1 = self.score_aux1(layer_3)
                score_aux2 = self.score_aux2(layer_2)
                score_aux3 = self.score_aux3(layer_1)
            elif self.cfg.WHICH_SCORE == 'up':
                score_aux1 = self.score_aux1(up1)
                score_aux2 = self.score_aux2(up2)
                score_aux3 = self.score_aux3(up3)

            score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux1
            score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux2
            score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux3

            result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            if cal_loss:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


class FCN_Conc_MultiModalTarget(nn.Module):

    def __init__(self, cfg, device=None):
        super(FCN_Conc_MultiModalTarget, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "./initmodel/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.head = _FCNHead(512, num_classes, nn.BatchNorm2d)
        # self.head = nn.Conv2d(512, num_classes, 1)

        if self.trans:
            self.build_upsample_content_layers(dims)

        self.score_aux1 = nn.Sequential(
            nn.Conv2d(dims[3] * 2, num_classes, 1)
        )

        self.score_aux2 = nn.Sequential(
            nn.Conv2d(dims[2] * 2, num_classes, 1)
        )
        self.score_aux3 = nn.Sequential(
            nn.Conv2d(dims[1] * 2, num_classes, 1)
        )

        if pretrained:
            init_weights(self.head, 'normal')

            if self.trans:
                init_weights(self.up1_depth, 'normal')
                init_weights(self.up2_depth, 'normal')
                init_weights(self.up3_depth, 'normal')
                init_weights(self.up4_depth, 'normal')
                init_weights(self.up1_seg, 'normal')
                init_weights(self.up2_seg, 'normal')
                init_weights(self.up3_seg, 'normal')
                init_weights(self.up4_seg, 'normal')


            init_weights(self.score_aux3, 'normal')
            init_weights(self.score_aux2, 'normal')
            init_weights(self.score_aux1, 'normal')
            init_weights(self.head, 'normal')

        else:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        if 'bottleneck' in self.cfg.FILTERS:
            self.up1_depth = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up2_depth = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
            self.up3_depth = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
            self.up4_depth = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)

            self.up1_seg = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up2_seg = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
            self.up3_seg = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
            self.up4_seg = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
        else:
            self.up1_depth = Conc_Up_Residual(dims[4], dims[3], norm=norm)
            self.up2_depth = Conc_Up_Residual(dims[3], dims[2], norm=norm)
            self.up3_depth = Conc_Up_Residual(dims[2], dims[1], norm=norm)
            self.up4_depth = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)

            self.up1_seg = Conc_Up_Residual(dims[4], dims[3], norm=norm)
            self.up2_seg = Conc_Up_Residual(dims[3], dims[2], norm=norm)
            self.up3_seg = Conc_Up_Residual(dims[2], dims[1], norm=norm)
            self.up4_seg = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)

        self.up_depth = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        self.up_seg = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target_1=None, target_2=None, label=None, phase='train', content_layers=None, cal_loss=True):
        result = {}
        layer_0 = self.relu(self.bn1(self.conv1(source)))
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        if self.trans:
            # content model branch
            up1_depth = self.up1_depth(layer_4, layer_3)
            up2_depth = self.up2_depth(up1_depth, layer_2)
            up3_depth = self.up3_depth(up2_depth, layer_1)
            up4_depth = self.up4_depth(up3_depth)
            result['gen_depth'] = self.up_depth(up4_depth)

            up1_seg = self.up1_seg(layer_4, layer_3)
            up2_seg = self.up2_seg(up1_seg, layer_2)
            up3_seg = self.up3_seg(up2_seg, layer_1)
            up4_seg = self.up4_seg(up3_seg)
            result['gen_seg'] = self.up_seg(up4_seg)

            if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_content_depth'] = self.content_model(result['gen_depth'], target_1, layers=content_layers)
                result['loss_content_seg'] = self.content_model(result['gen_seg'], target_2, layers=content_layers)

        if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.INFERENCE:

            score_head = self.head(layer_4)

            score_aux1 = None
            score_aux2 = None
            score_aux3 = None
            if self.cfg.WHICH_SCORE == 'main':
                score_aux1 = self.score_aux1(layer_3)
                score_aux2 = self.score_aux2(layer_2)
                score_aux3 = self.score_aux3(layer_1)
            elif self.cfg.WHICH_SCORE == 'up':

                score_aux1 = self.score_aux1(torch.cat((up1_depth, up1_seg), 1))
                score_aux2 = self.score_aux2(torch.cat((up2_depth, up2_seg), 1))
                score_aux3 = self.score_aux3(torch.cat((up3_depth, up3_seg), 1))

            score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux1
            score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux2
            score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux3

            result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            if cal_loss:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)

        # if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
        #     result['loss_pix2pix_depth'] = self.pix2pix_criterion(result['gen_depth'], target_1)
        #     result['loss_pix2pix_seg'] = self.pix2pix_criterion(result['gen_seg'], target_2)

        return result



class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, padding=0, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


#######################################################################
class UNet(nn.Module):
    def __init__(self, cfg, device=None):
        super(UNet, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "./initmodel/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

        self.score = nn.Conv2d(dims[1], num_classes, 1)

        # norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
        self.up1 = Conc_Up_Residual(dims[4], dims[3], norm=nn.BatchNorm2d)
        self.up2 = Conc_Up_Residual(dims[3], dims[2], norm=nn.BatchNorm2d)
        self.up3 = Conc_Up_Residual(dims[2], dims[1], norm=nn.BatchNorm2d)
        self.up4 = Conc_Up_Residual(dims[1], dims[1], norm=nn.BatchNorm2d, conc_feat=False)

        if pretrained:
            init_weights(self.up1, 'normal')
            init_weights(self.up2, 'normal')
            init_weights(self.up3, 'normal')
            init_weights(self.up4, 'normal')
            init_weights(self.score, 'normal')

        else:

            init_weights(self, 'normal')

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def forward(self, source=None, label=None):
        result = {}

        layer_1 = self.layer1(self.relu(self.bn1(self.conv1(source))))
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        up1 = self.up1(layer_4, layer_3)
        up2 = self.up2(up1, layer_2)
        up3 = self.up3(up2, layer_1)
        up4 = self.up4(up3)

        result['cls'] = self.score(up4)
        result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):

    def __init__(self, cfg, batch_norm, bins=(1, 2, 3, 6), dropout=0.1,
                 zoom_factor=8, use_ppm=True, pretrained=True, device=None):
        super(PSPNet, self).__init__()
        assert 2048 % len(bins) == 0
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        # models.BatchNorm = batch_norm
        self.BatchNorm = batch_norm
        self.device = device
        self.trans = not cfg.NO_TRANS
        self.cfg = cfg
        dims = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        if self.trans:
            self.build_upsample_content_layers(dims)

        resnet = models.__dict__[cfg.ARCH](pretrained=pretrained, deep_base=True)
        print("load ", cfg.ARCH)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins, self.BatchNorm)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            self.BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, cfg.NUM_CLASSES, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                self.BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, cfg.NUM_CLASSES, kernel_size=1)
            )

        if self.trans:
            init_weights(self.up1, 'normal')
            init_weights(self.up2, 'normal')
            init_weights(self.up3, 'normal')
            init_weights(self.up4, 'normal')
            init_weights(self.cross_1, 'normal')
            init_weights(self.cross_2, 'normal')
            init_weights(self.cross_3, 'normal')
            init_weights(self.up_seg, 'normal')

        init_weights(self.aux, 'normal')
        init_weights(self.cls, 'normal')
        init_weights(self.ppm, 'normal')

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
        # norm = self.norm
        self.cross_1 = nn.Conv2d(dims[7], dims[4], kernel_size=1, bias=False)
        self.cross_2 = nn.Conv2d(dims[6], dims[3], kernel_size=1, bias=False)
        self.cross_3 = nn.Conv2d(dims[5], dims[3], kernel_size=1, bias=False)
        self.up1 = Conc_Residual_bottleneck(dims[5], dims[4], norm=norm)
        self.up2 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
        self.up3 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
        self.up4 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)

        self.up_seg = nn.Sequential(
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        self.score_head = nn.Conv2d(512, self.cfg.NUM_CLASSES, 1)
        self.score_aux1 = nn.Conv2d(256, self.cfg.NUM_CLASSES, 1)
        self.score_aux2 = nn.Conv2d(128, self.cfg.NUM_CLASSES, 1)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def set_content_model(self, content_model):
        self.content_model = content_model.to(self.device)

    def forward(self, source, target=None, label=None, phase='train', content_layers=None, cal_loss=True, matrix=None):

        x = source
        y = label
        result = {}
        # x_size = x.size()
        # assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        # h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        # w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        layer_0 = self.layer0(x)
        layer_1 = self.layer1(self.maxpool(layer_0))
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)
        # print('x', x.size())
        # print('layer_0', layer_0.size())
        # print('layer_1', layer_1.size())
        # print('layer_2', layer_2.size())
        # print('layer_3', layer_3.size())
        # print('layer_4', layer_4.size())

        # print(x.size())

        x = layer_4
        if self.use_ppm:
            x = self.ppm(x)
        if not self.trans or phase=='test':
            x = self.cls(x)
            if self.zoom_factor != 1:
                result['cls'] = F.interpolate(x, size=source.size()[2:], mode='bilinear', align_corners=True)

            if cal_loss:
                aux = self.aux(layer_3)
                if self.zoom_factor != 1:
                    aux = F.interpolate(aux, size=source.size()[2:], mode='bilinear', align_corners=True)
                main_loss = self.cls_criterion(result['cls'], y)
                aux_loss = self.cls_criterion(aux, y)
                result['loss_cls'] = main_loss + 0.4 * aux_loss

        else:
            cross_1 = self.cross_1(x)
            cross_2 = self.cross_2(layer_4)
            cross_3 = self.cross_3(layer_3)
            cross_conc = torch.cat((cross_1, cross_2, cross_3), 1)
            up1_seg = self.up1(cross_conc, layer_2)
            up2_seg = self.up2(up1_seg, layer_1)
            up3_seg = self.up3(up2_seg, layer_0)
            up4_seg = self.up4(up3_seg)

            result['gen_img'] = self.up_seg(up4_seg)

            score_head = self.score_head(up1_seg)
            score_aux1 = self.score_aux1(up2_seg)
            score_aux2 = self.score_aux2(up3_seg)

            x = self.cls(x)
            score = F.interpolate(x, score_head.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_head
            score = F.interpolate(score, score_aux1.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux1
            score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux2
            result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            if cal_loss:
                main_loss = self.cls_criterion(result['cls'], y)
                result['loss_cls'] = main_loss

                aux = self.aux(layer_3)
                if self.zoom_factor != 1:
                    aux = F.interpolate(aux, size=source.size()[2:], mode='bilinear', align_corners=True)
                main_loss = self.cls_criterion(result['cls'], y)
                aux_loss = self.cls_criterion(aux, y)
                result['loss_cls'] = main_loss + 0.4 * aux_loss
                result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

        # if matrix is not None:
        #     prediction_matrix = torch.zeros(source.size()[0], self.cfg.NUM_CLASSES, self.cfg.FINE[0],
        #                                     self.cfg.BASE_SIZE[1]).to(self.device)
        #     count_crop_matrix = torch.zeros(source.size()[0], 1, self.cfg.BASE_SIZE[0], self.cfg.BASE_SIZE[1]).to(
        #         self.device)
        #     prediction_matrix[:source.size()[0], matrix[0]:matrix[1],matrix[2]:matrix[3]] += result['cls']
        #     count_crop_matrix[:source.size()[0], matrix[0]:matrix[1],matrix[2]:matrix[3]] += 1
        #     result['matrix_pred'] = prediction_matrix
        #     result['matrix_count'] = count_crop_matrix

        return result

###########################################################################################
###########################################################################################
##########################                                 ################################
##########################        con                      ################################
##########################                                 ################################
###########################################################################################
class Encoder(nn.Module):

    def __init__(self, encoder='resnet18', pretrained='imagenet',in_channel=1):
        super(Encoder, self).__init__()
        # if pretrained == 'imagenet' or pretrained == 'place':
        #     is_pretrained = True
        # else:
        is_pretrained = False

        # if pretrained == 'place':
        #     resnet = models.__dict__[encoder](num_classes=365)
        #     load_path = '/home/dudapeng/workspace/pretrained/place/' + encoder + '_places365.pth'
        #     checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
        #     state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        #     resnet.load_state_dict(state_dict)
        #     print('place {0} loaded....'.format(encoder))
        # else:
        #     resnet = models.__dict__[encoder](pretrained=is_pretrained)
        #     print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
        resnet = models.__dict__[encoder](pretrained=False)


        self.conv1 = resnet.conv1
        # self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(512*2*2, 64)
        )

        dims = [32, 64, 128, 256, 512, 1024, 2048]
        norm = nn.InstanceNorm2d
        self.up1 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
        self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
        self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
        self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm)
        self.up5 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)

        self.up_l_14 = nn.Sequential(
            nn.Conv2d(dims[3], 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        self.up_l_28 = nn.Sequential(
            nn.Conv2d(dims[2], 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.up_l_56 = nn.Sequential(
            nn.Conv2d(dims[1], 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.up_l_112 = nn.Sequential(
            nn.Conv2d(dims[1], 1, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        self.up_l_224 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 1, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        self.up_ab_14 = nn.Sequential(
            nn.Conv2d(dims[3], 2, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        self.up_ab_28 = nn.Sequential(
            nn.Conv2d(dims[2], 2, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.up_ab_56 = nn.Sequential(
            nn.Conv2d(dims[1], 2, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.up_ab_112 = nn.Sequential(
            nn.Conv2d(dims[1], 2, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        self.up_ab_224 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 2, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        if not is_pretrained:
            init_weights(self, 'normal')

    def forward(self, x, no_grad=False):
        out = {}

        if no_grad:
            with torch.no_grad():
                layer_0 = self.relu(self.bn1(self.conv1(x)))
                layer_1 = self.maxpool(self.layer1(layer_0))
                layer_2 = self.layer2(layer_1)
                out['feat_128'] = layer_2
                layer_3 = self.layer3(layer_2)
                out['feat_256'] = layer_3
                layer_4 = self.layer4(layer_3)
                out['feat_512'] = layer_4
                out['z'] = self.fc(layer_4)
        else:
            layer_0 = self.relu(self.bn1(self.conv1(x)))
            layer_1 = self.maxpool(self.layer1(layer_0))
            layer_2 = self.layer2(layer_1)
            out['feat_128'] = layer_2
            layer_3 = self.layer3(layer_2)
            out['feat_256'] = layer_3
            layer_4 = self.layer4(layer_3)
            out['feat_512'] = layer_4
            # print(layer_4.size())
            out['z'] = self.fc(layer_4)



            L_gen = []
            AB_gen = []


            up1 = self.up1(layer_4, layer_3)
            up2 = self.up2(up1, layer_2)
            up3 = self.up3(up2, layer_1)
            up4 = self.up4(up3, layer_0)
            up5 = self.up5(up4)

            l_14 = self.up_l_14(up1)
            l_28 = self.up_l_28(up2)
            l_56 = self.up_l_56(up3)
            l_112 = self.up_l_112(up4)
            l_224 = self.up_l_224(up5)

            ab_14 = self.up_ab_14(up1)
            ab_28 = self.up_ab_28(up2)
            ab_56 = self.up_ab_56(up3)
            ab_112 = self.up_ab_112(up4)
            ab_224 = self.up_ab_224(up5)


            L_gen.append(l_224)
            L_gen.append(l_112)
            L_gen.append(l_56)
            L_gen.append(l_28)
            L_gen.append(l_14)

            AB_gen.append(ab_224)
            AB_gen.append(ab_112)
            AB_gen.append(ab_56)
            AB_gen.append(ab_28)
            AB_gen.append(ab_14)

            out['L'] = L_gen
            out['AB'] = AB_gen


        return out


class Contrastive_CrossModal_Conc(nn.Module):

    def __init__(self, cfg, alpha=0.5, beta=1.0, gamma=0.05, feat_channel=128, feat_size=8, device=None):
        super(Contrastive_CrossModal_Conc, self).__init__()

        self.cfg = cfg
        self.device = device
        self.encoder_rgb = Encoder(pretrained='')
        self.evaluator_rgb = Evaluator(cfg.NUM_CLASSES)
        # self.avg_pool = nn.AvgPool2d(7, 1)
        self.avg_pool = nn.AvgPool2d(2, 1)

        # self.z = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(1024 * 7 * 7, 64)
        # )

        self.prior_d_cross = PriorDiscriminator(64)
        self.local_d_inner_rgb = LocalDiscriminator(feat_channel + 64)
        self.local_l_cross = LocalDiscriminator(2)  # + 128
        self.local_ab_cross = LocalDiscriminator(4)  # + 128


        # self.cls_criterion = torch.nn.CrossEntropyLoss(cfg.CLASS_WEIGHTS_TRAIN)
        self.cls_criterion = torch.nn.CrossEntropyLoss()

        self.pixel_criterion = torch.nn.L1Loss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.feat_size = feat_size

        init_weights(self, 'normal')
        #
        # init_weights(self, 'normal')
        # init_weights(self, 'normal')
        # init_weights(self, 'normal')
        # init_weights(self, 'normal')
        # init_weights(self, 'normal')

    def forward(self, source, target=None, label=None, class_only=False):
        rgb=source

        res_dict = {}
        if class_only:
            # shortcut to encode one image and evaluate classifier
            result_rgb = self.encoder_rgb(rgb, no_grad=True)
            avg_rgb = self.avg_pool(result_rgb['feat_512'])
            lgt_glb_mlp_rgb, lgt_glb_lin_rgb = self.evaluator_rgb(avg_rgb)
            res_dict['class_rgb'] = [lgt_glb_mlp_rgb, lgt_glb_lin_rgb]
            return res_dict
        l,ab=[],[]
        for i in range(self.cfg.MULTI_SCALE_NUM):
            _l,_ab = torch.split(target[i],[1,2],dim=1)
            l.append(_l)
            ab.append(_ab)

        result_rgb = self.encoder_rgb(rgb)
        avg_rgb = self.avg_pool(result_rgb['feat_512'])
        lgt_glb_mlp_rgb, lgt_glb_lin_rgb = self.evaluator_rgb(avg_rgb)

        z_rgb = result_rgb['z']
        f_rgb = result_rgb['feat_128']

        z_rgb_exp = z_rgb.unsqueeze(-1).unsqueeze(-1)
        z_rgb_exp = z_rgb_exp.expand(-1, -1, self.feat_size, self.feat_size)

        prior = torch.rand_like(z_rgb)

        term_a = torch.log(self.prior_d_cross(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d_cross(z_rgb)).mean()
        PRIOR_RGB = - (term_a + term_b) * self.gamma
        inner_neg_rgb = torch.cat((f_rgb[1:], f_rgb[0].unsqueeze(0)), dim=0)
        # print(f_rgb.size())
        # print(z_rgb_exp.size())
        inner_y_M_RGB = torch.cat((f_rgb, z_rgb_exp), dim=1)


        inner_y_M_prime_RGB = torch.cat((inner_neg_rgb, z_rgb_exp), dim=1)

        Ej = -F.softplus(-self.local_d_inner_rgb(inner_y_M_RGB)).mean()
        Em = F.softplus(self.local_d_inner_rgb(inner_y_M_prime_RGB)).mean()
        LOCAL_RGB = (Em - Ej) * self.beta

        LOCAL_cross_ab = torch.zeros(1).to(self.device)
        LOCAL_cross_l = torch.zeros(1).to(self.device)

        for i, (L_gen,AB_gen,L_label, AB_label) in enumerate(zip(result_rgb['L'],result_rgb['AB'],l,ab)):

            if i + 1 > self.cfg.MULTI_SCALE_NUM:
                break

            cross_pos = torch.cat((L_gen, L_label), 1)
            cross_neg = torch.cat((L_gen, torch.cat((L_label[1:], L_label[0].unsqueeze(0)), dim=0)), 1)
            Ej = -F.softplus(-self.local_l_cross(cross_pos)).mean()
            Em = F.softplus(self.local_l_cross(cross_neg)).mean()
            LOCAL_cross_l += (Em - Ej) * self.beta

            cross_pos = torch.cat((AB_gen, AB_label), 1)
            cross_neg = torch.cat((AB_gen, torch.cat((AB_label[1:], AB_label[0].unsqueeze(0)), dim=0)), 1)
            Ej = -F.softplus(-self.local_ab_cross(cross_pos)).mean()
            Em = F.softplus(self.local_ab_cross(cross_neg)).mean()
            LOCAL_cross_ab += (Em - Ej) * self.beta
        # cls_loss = self.cls_criterion(lgt_glb_mlp_rgb, label) + self.cls_criterion(lgt_glb_lin_rgb, label)

        # pixel_loss = self.pixel_criterion(result_rgb['ms_gen'][0], depth[0])

        # return {'cls_loss': cls_loss, 'pixel_loss': pixel_loss, 'ms_gen': result_rgb['ms_gen']}
        return {'gen_l_loss': sum(LOCAL_cross_l),'gen_ab_loss': sum(LOCAL_cross_ab), 'local_rgb_loss': LOCAL_RGB, 
                'prior_rgb_loss': PRIOR_RGB, 'l_gen': result_rgb['L'], 'ab_gen': result_rgb['AB']}


class GlobalDiscriminator(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.c0 = nn.Conv2d(in_channel, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.l0 = nn.Linear(32 * 10 * 10 + 128, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.c0 = nn.Conv2d(in_channel, 64, kernel_size=1)
        self.c1 = nn.Conv2d(64, 128, kernel_size=1)
        self.c2 = nn.Conv2d(128, 256, kernel_size=1)
        self.c3 = nn.Conv2d(256, 512, kernel_size=1)
        self.c4 = nn.Conv2d(512, 1, kernel_size=1)

        init_weights(self, 'normal')

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        return self.c4(h)


class PriorDiscriminator(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.l0 = nn.Linear(in_channel, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

        init_weights(self, 'normal')

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 15)
        self.bn1 = nn.BatchNorm1d(15)
        self.l2 = nn.Linear(15, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.l3 = nn.Linear(10, 10)
        self.bn3 = nn.BatchNorm1d(10)

    def forward(self, x):
        encoded, _ = x[0], x[1]
        clazz = F.relu(self.bn1(self.l1(encoded)))
        clazz = F.relu(self.bn2(self.l2(clazz)))
        clazz = F.softmax(self.bn3(self.l3(clazz)), dim=1)
        return clazz

class Evaluator(nn.Module):
    def __init__(self, n_classes):
        super(Evaluator, self).__init__()
        self.n_classes = n_classes
        self.block_glb_mlp = \
            MLPClassifier(512, self.n_classes, n_hidden=1024, p=0.2)
        self.block_glb_lin = \
            MLPClassifier(512, self.n_classes, n_hidden=None, p=0.0)

    def forward(self, ftr_1):
        '''
        Input:
          ftr_1 : features at 1x1 layer
        Output:
          lgt_glb_mlp: class logits from global features
          lgt_glb_lin: class logits from global features
        '''
        # collect features to feed into classifiers
        # - always detach() -- send no grad into encoder!
        h_top_cls = flatten(ftr_1).detach()
        # h_top_cls = flatten(ftr_1)
        # compute predictions
        lgt_glb_mlp = self.block_glb_mlp(h_top_cls)
        lgt_glb_lin = self.block_glb_lin(h_top_cls)
        return lgt_glb_mlp, lgt_glb_lin


class MLPClassifier(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super(MLPClassifier, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True)
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True)
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)

def flatten(x):
    return x.reshape(x.size(0), -1)

