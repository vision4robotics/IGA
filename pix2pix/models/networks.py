from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):

    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'search':
        net = PB()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.r = nn.LeakyReLU(negative_slope=0.2)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.r(x)
        x = self.bn(x)
        return x

class SmallBranch(nn.Module):
    def __init__(self):
        super(SmallBranch, self).__init__()

        self.conv3 = nn.Sequential(BasicConv2d(3, 64, kernel_size=(1, 3), padding=(0, 1)),
                                   BasicConv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)))
        self.conv5 = nn.Sequential(BasicConv2d(64, 32, kernel_size=(1, 5), padding=(0, 2)),
                                   BasicConv2d(32, 32, kernel_size=(5, 1), padding=(2, 0)))
        self.conv8 = nn.Sequential(BasicConv2d(32, 32, kernel_size=(1, 8), stride=2, padding=(0, 3)),
                                   BasicConv2d(32, 32, kernel_size=(8, 1), stride=2, padding=(3, 0)))

    def forward(self, x):
        out = self.conv3(x)
        out = self.conv5(out)
        out = self.conv8(out)
        return out

class MiddleBranch(nn.Module):
    def __init__(self):
        super(MiddleBranch, self).__init__()

        self.conv4_1 = nn.Sequential(BasicConv2d(3, 64, kernel_size=(1, 4), padding=(0, 0)),
                                     BasicConv2d(64, 64, kernel_size=(4, 1), padding=(0, 0)))
        self.conv4_2 = nn.Sequential(BasicConv2d(64, 32, kernel_size=(1, 4), stride=2, padding=(0, 2)),
                                     BasicConv2d(32, 32, kernel_size=(4, 1), stride=2, padding=(2, 0)))

    def forward(self, x):
        out = self.conv4_1(x)
        out = self.conv4_2(out)
        return out

class LargeBranch(nn.Module):
    def __init__(self):
        super(LargeBranch, self).__init__()

        self.conv12 = nn.Sequential(BasicConv2d(3, 32, kernel_size=(1, 11), stride=2, padding=(0, 5)),
                                    BasicConv2d(32, 32, kernel_size=(11, 1), stride=2, padding=(5, 0)))

    def forward(self, x):
        out = self.conv12(x)
        return out

class SemanticFeature(nn.Module):
    def __init__(self):
        super(SemanticFeature, self).__init__()

        self.conv1 = BasicConv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.s = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm2d(num_features=3)
    def forward(self, f1, f2, f3):
        f1 = self.conv1(f1)
        f2 = self.conv1(f2)
        f3 = self.conv1(f3)
        c = torch.cat([f1, f2, f3], dim=1)
        fe = self.s(c)
        fe = self.bn(fe)
        return fe

class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.conv = BasicConv2d(3, 32, kernel_size=1, stride=1, padding=0)
        self.conv3 = BasicConv2d(32, 32, kernel_size=4, stride=4, padding=0)
        #self.conv3 = BasicConv2d(32, 32, kernel_size=2, stride=2, padding=0)
        self.residual = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(32, eps=0.001)
                                      )

    def forward(self, f1, f2, f3, fe):
        fe = self.conv(fe)
        f1_e = torch.mul(f1,fe)
        f2_e = torch.mul(f2,fe)
        f3_e = torch.mul(f3,fe)
        fe_residual = self.residual(fe)
        sum = torch.add(f1_e, f2_e)
        sum = torch.add(sum, f3_e)
        sum = torch.add(sum, fe_residual)
        out = self.conv3(sum)
        #out = self.conv3(out)
        #print("out:",out)
        #print("out.shape:", out.shape)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.Small = SmallBranch()
        self.Middle = MiddleBranch()
        self.Large = LargeBranch()
        self.semantic = SemanticFeature()
        self.Output = FeatureFusion()

    def forward(self, x):
        f1 = self.Small(x)
        f2 = self.Middle(x)
        f3 = self.Large(x)
        fe = self.semantic(f1, f2, f3)
        out = self.Output(f1, f2, f3, fe)

        return out

#############################################
############Transformer Generator############
#############################################


from .config import cfg
from .utile import TFGenerater

# test = TFGenerater(cfg).cuda()
# print(torch.cuda.is_available)
# y = test(k.cuda())
# test.train()
# test.eval()

class InitPerturbationGenerator(nn.Module):
    def __init__(self):
        super(InitPerturbationGenerator, self).__init__()

        self.encoder = Encoder()
        self.generator = TFGenerater(cfg).cuda()


    def forward(self, x):
        F = self.encoder(x)
        I_Perturbation = self.generator(F.cuda())
        #print("I_Perturbation.shape:", I_Perturbation.shape)
        return I_Perturbation, F

class feature_fusion(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(feature_fusion, self).__init__()
        self.compress = nn.Conv2d(ch_in, 3, kernel_size=1, stride=1)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, ch_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature, p_0):
        feature = self.compress(feature)
        #print("compress:",feature)
        f_fusion = feature + p_0
        #print("f_fusion:", f_fusion)
        f_fusion = self.conv(f_fusion)
        #print("f_fusion_2:", f_fusion)

        return f_fusion


class para_infer(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(para_infer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, ch_out, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        feature = self.conv(x)
        para = torch.sigmoid(feature)
        #print("para:", para)
        return para


class iter_esti(nn.Module):
    def __init__(self, ch_in):
        super(iter_esti, self).__init__()
        self.index = {
            '0': 4,
            '1': 5,
            '2': 6,
            '3': 7,
            '4': 8,
            '5': 9,
        }
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.ie_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        feature = self.conv(x)
        iter = self.ie_net(feature.squeeze())

        iter = torch.softmax(iter, dim= 0)

        iter = torch.argmax(iter, dim=0).cpu().numpy()+1
        #print("iter:",np.max(iter))
        #iter = self.index[str(iter)]
        return np.max(iter)


class perturbation_augument(nn.Module):
    def __init__(self, ch_in):
        super(perturbation_augument, self).__init__()
        #self.fusion = feature_fusion(ch_in, 32)
        self.para = para_infer(3, 3)
        self.iter = iter_esti(32)
        self.bn = nn.BatchNorm2d(num_features=3)
    def forward(self, x, p_0):
        para = self.para(p_0)
        iter = self.iter(x)
        for i in range(iter):
            p_aug = (1 + para) * p_0 - torch.pow(p_0, 2)
        #print("p_aug:",p_aug)
        p_aug = self.bn(p_aug)
        #print("p_aug_bn:", p_aug)
        return p_aug

class PB(nn.Module):
    def __init__(self):
        super(PB, self).__init__()

        self.initP = InitPerturbationGenerator()
        self.enhance = perturbation_augument(32)
        self.bn = nn.BatchNorm2d(num_features=3)
        self.criterionL2 = torch.nn.MSELoss()
    def forward(self, x, x_255):
        #print("input x:", x)
        P0, F = self.initP(x)
        #print("P0:",P0)
        P0 = self.bn(P0)/30

        x_middle = P0 + x_255

        #print("P0_norm [0 1]:", P0)
        #print("F:",F)
        Final_Perturbation = self.enhance(F, P0)
        return x_middle, Final_Perturbation