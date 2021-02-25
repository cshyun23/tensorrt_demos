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
import math
import sys
sys.path.append("../lib")
import time

# import _init_paths
import models
from config import cfg
from config import update_config
from utils.transforms import transform_preds
from utils.transforms import get_affine_transform

####eval-related###########
import pycocotools.coco
import pycocotools.cocoeval
import json
###########################

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

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals



def get_final_preds(batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if True:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals

def get_person_detection_boxes(imgId, person_detection_list):

    tmp = []
    for detection_result in person_detection_list:
        if int(imgId) == detection_result['image_id']:
            #print(detection_result)
            tmp.append(detection_result)

    pred_boxes=[]
    for detection_result in tmp:
        pred_class, pred_box, pred_score = detection_result['category_id'], detection_result['bbox'], detection_result['score']
        #print(pred_class, pred_box, pred_score)
        if (pred_class == 1) and (pred_score > 0.9) :
            top_left_x = int(pred_box[0])
            top_left_y = int(pred_box[1])
            bottom_right_x = top_left_x + int(pred_box[2])
            bottom_right_y = top_left_y + int(pred_box[3])
            temp = [(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)]
            #print(temp)
            pred_boxes.append(temp)

    return pred_boxes
    

def get_pose_estimation_prediction(ort_session, image, centers, scales, transform):
    rotation = 0
    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)
            
        #cv2.imwrite("crop.png", model_input)

        # hwc -> 1chw
        model_input = transform(model_input)#.unsqueeze(0)
        model_inputs.append(model_input)

    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)
    
    ort_in = {ort_session.get_inputs()[0].name: to_numpy(model_inputs)}

    ort_out = ort_session.run(None, ort_in)
    output = ort_out[0]
    
    #print(len(ort_out), len(ort_out[0]), len(ort_out[0][0]), len(ort_out[0][0][0]), len(ort_out[0][0][0][0]))

    # compute output heatmap
    coords, _ = get_final_preds(
        output,
        np.asarray(centers),
        np.asarray(scales))

    return coords


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int
    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--writeBoxFrames', action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)

    #box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #box_model.to(CTX)
    #box_model.eval()
    
    ############ onnx 변환 #############################
    onnx_model_name = "save/pose_hourglass_1_256x192.onnx"
    
    if os.path.exists(onnx_model_name) == False:
        pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=False
        )

        if cfg.TEST.MODEL_FILE:
            print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
            pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
        else:
            print('expected model defined in config at TEST.MODEL_FILE')
        pose_model.to(CTX)
        pose_model.eval()
    
        x = torch.ones((2,3,256,192)).cuda()
        torch_out = pose_model(x)
    
        torch.onnx.export(pose_model,
    			x,
    			onnx_model_name,
    			export_params=True,
    			opset_version=11,
    			do_constant_folding=True,
    			input_names = ['input'],
    			output_names = ['output'],
    			dynamic_axes={'input' : {0 : 'batch_size'},
    					'output' : {0 : 'batch_size'}})
    
    import onnx
    
    onnx_model = onnx.load(onnx_model_name)
    #onnx.checker.check_model(onnx_model)
    
    import onnxruntime
    
    ort_session = onnxruntime.InferenceSession(onnx_model_name)
    ###################################################

    #person box
    person_detect_file = open('COCO_val2017_detections_AP_H_56_person.json', 'r')
    person_detect_list = json.load(person_detect_file)

    # validation set of images
    images_dir = '/home/golf2/deep-high-resolution-net.pytorch/data/val2017'
    annotation_file = '/home/golf2/deep-high-resolution-net.pytorch/data/annotations_trainval2017/annotations/person_keypoints_val2017.json'
    cocoGt = pycocotools.coco.COCO(annotation_file)
    catIds = cocoGt.getCatIds('person')
    imgIds = cocoGt.getImgIds()
    
    #imgIds = cocoGt.getImgIds(catIds=catIds)
    
    results = []
    for n, imgId in enumerate(imgIds):
        
        #read image
        
        img = cocoGt.imgs[imgId]
        img_path = os.path.join(images_dir, img['file_name'])
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        #print(imgId)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Clone 2 image for person detection and pose estimation
        if cfg.DATASET.COLOR_RGB:
            image_per = image_rgb.copy()
            image_pose = image_rgb.copy()
        else:
            image_per = image_bgr.copy()
            image_pose = image_bgr.copy()

        # Clone 1 image for debugging purpose
        image_debug = image_bgr.copy()

        # object detection box
        #pred_boxes = get_person_detection_boxes(imgId, image_per, box_model, threshold=0.9)
        pred_boxes = get_person_detection_boxes(imgId, person_detect_list)


        # Can not find people. Move to next image
        if not pred_boxes:
            continue
        
        
        for box in pred_boxes:
            cv2.rectangle(image_debug, box[0], box[1], color=(0, 255, 0), thickness=3)  # Draw Rectangle with the coordinates
        
        # pose estimation : for multiple people
        centers = []
        scales = []
        for box in pred_boxes:
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            centers.append(center)
            scales.append(scale)

        pose_preds = get_pose_estimation_prediction(ort_session, image_pose, centers, scales, transform=pose_transform)


        for coords in pose_preds:
            # Draw each point on image
            score = 0.0
            kps = [0]*(17*3)
            #print(coords.shape)
            for i, coord in enumerate(coords):
                
                x_coord, y_coord = int(coord[0]), int(coord[1])
                cv2.circle(image_debug, (x_coord, y_coord), 4, (255, 0, 0), 2)
                score += 1.0
                kps[i * 3 + 0] = x_coord
                kps[i * 3 + 1] = y_coord
                kps[i * 3 + 2] = 2
                
            ann = {
                'image_id': imgId,
                'category_id': 1,
                'keypoints': kps,
                'score': score / 17.0
                }
            results.append(ann)
                
        '''
        cv2.imshow("pos", image_debug)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        '''

        if n %100 == 0:
            print('%d / %d' % (n, len(imgIds)))

    #cv2.destroyAllWindows()
    with open(onnx_model_name+'_results.json', 'w') as f:
        json.dump(results, f)
        
    cocoDt = cocoGt.loadRes(onnx_model_name+'_results.json')
    cocoEval = pycocotools.cocoeval.COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.params.imgIds = imgIds
    
    #cocoEval.params.catIds = [1]
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    main()
