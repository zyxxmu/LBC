# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
# from engine import train_one_epoch_multi_reso,train_one_epoch_multi_reso_distill, evaluate_multi_reso_in_train, evaluate_multi_reso_in_eval,evaluate
from engine import *
from losses import DistillationLoss
from samplers import RASampler
import models
import utils
import shutil
import torchvision
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str

import pdb


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_deit_small_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--multi-reso', default=False, action='store_true',help='')
    parser.add_argument('--use-kl', default=False, action='store_true',help='')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    return parser



class NativeScaler_multi_reso:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        # self._scaler.step(optimizer)
        # self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



def main(args):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
    )
    model = model.cuda()
    # for size in args.input_size_list:
    #     model.apply(lambda m: setattr(m, 'current_img_size', size))
    #     input = torch.randn((1,3,size,size))
    #     flops = FlopCountAnalysis(model, input)
    #     print(f'{size}:{flops.total()}')
    #     print(flop_count_table(flops, max_depth=4))
    #     pdb.set_trace()
    input = torch.randn((1,3,224,224)).cuda()
    flops = FlopCountAnalysis(model, input)
    print(flop_count_table(flops, max_depth=4))

  




if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # args.input_size_list.sort(reverse=True)
    main(args)
