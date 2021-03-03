from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np

import time

# import _init_paths
import models
from config import cfg
from config import update_config
from utils.transforms import get_affine_transform

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def parse_args():
    parser = argparse.ArgumentParser(description='pytorch to onnx')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--outputDir', type=str, required=True)

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--pr',
                        help='pruning ratio',
                        type=float,
                        default=0.5)

    args = parser.parse_args()
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    pruning_ratio = args.pr
    '''
    pose_model = eval('models.'+cfg.MODEL.NAME+'_prun.get_pose_net')(
        cfg, pruning_ratio, is_train=False
    )
    '''
    pose_model = eval('models.'+cfg.MODEL.NAME+'_prun_v2.get_pose_net')(
        cfg, is_train=False
    )
    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    ############ onnx 변환 #############################
    pose_model_name = cfg.TEST.MODEL_FILE.split('/')[-1].split('.')[0]
    input_dim = pose_model_name.split('_')[-1]
    if 'x' in input_dim:
        dim_split = input_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad input_dim (%s)!' % input_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else :
        print("W, H parsing error")
    #print(pose_model_name)
    #print(w,h)

    onnx_model_path = os.path.join(args.outputDir, pose_model_name + ".onnx")
    x = torch.ones((1,3,w,h)).cuda()
    torch_out = pose_model(x)
    
    if os.path.exists(onnx_model_path) == False:
        torch.onnx.export(pose_model,
    			x,
    			onnx_model_path,
    			export_params=True,
    			opset_version=11,
    			do_constant_folding=True,
    			input_names = ['input'],
    			output_names = ['output'],
    			dynamic_axes={'input' : {0 : 'batch_size'},
    					'output' : {0 : 'batch_size'}})
    else :
        print(onnx_model_path, "already exists")

if __name__ == '__main__':
    main()
