import os
import time
import argparse
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
from utils.pose_with_plugins import TrtPOSE
from PIL import Image
import numpy as np
import math
from utils.transforms import transform_preds
from utils.transforms import get_affine_transform

####eval-related###########
import pycocotools.coco
import pycocotools.cocoeval
import json
###########################

JOINT_PAIR = {
	(15, 13),
	(13, 11),
	(11, 5),
	(16, 14),
	(14, 12),
	(12, 6),
	(5, 7),
	(7, 9),
	(5, 6),
	(6, 8),
	(8, 10),
	(3, 1),
	(1, 2),
	(2, 4),
	(1, 0),
	(0, 2)
}

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


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'POSE model on Jetson')
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '-v', '--video', type=str, default="golf-swing.mp4", 
        help='video file path')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='pose model path')

    args = parser.parse_args()

    return args


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

def loop_and_detect(model_name, video_file, trt_pose, input_shape, conf_th):

    #person box
    person_detect_file = open('COCO_val2017_detections_AP_H_56_person.json', 'r')
    person_detect_list = json.load(person_detect_file)

    #validation set of images
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


        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Clone 2 image for person detection and pose estimation
        
        image_per = image_rgb.copy()
        image_pose = image_rgb.copy()

        # Clone 1 image for debugging purpose
        image_debug = image_bgr.copy()

        ########### object detection box #######
        pred_boxes = get_person_detection_boxes(imgId, person_detect_list)
        #print(pred_boxes)

        # Can not find people. Move to next frame
        if not pred_boxes:
            continue

        for box in pred_boxes:
            #print(type(box[0][0]))
            #cv2.rectangle(image_debug, box[0], box[1], color=(0, 255, 0), thickness=3)  # Draw Rectangle with the coordinates
            pass

        # pose estimation : for multiple people
        for box in pred_boxes:
            center, scale = box_to_center_scale(box, input_shape[0], input_shape[1])

            centers=[center]
            scales=[scale]
            for center, scale in zip(centers, scales):
                trans = get_affine_transform(center, scale, 0, input_shape)
                # Crop smaller image of people
                input_img = cv2.warpAffine(
                    image_pose,
                    trans,
                    (input_shape[0], input_shape[1]),
                    flags=cv2.INTER_LINEAR)

            output = trt_pose.estimation(model_name, input_img)

            # compute output heatmap
            coords, _ = get_final_preds(
                output,
                np.asarray(centers),
                np.asarray(scales))

            pose_preds = coords

            for coords in pose_preds:
                # Draw each point on image
                score = 0.0
                kps = [0]*(17*3)
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

                # Draw line for each joints
                for pair_ab in JOINT_PAIR:
                    point_a = pair_ab[0]
                    point_b = pair_ab[1]
                    start_point = ( int( coords[point_a][0] ), int( coords[point_a][1] ) )
                    end_point   = ( int( coords[point_b][0] ), int( coords[point_b][1] ) )
                    color = (0, 0, 255)
                    cv2.line( image_debug, start_point, end_point, color, 2)
        
        if n % 100 == 0:
            print('%d / %d' % (n, len(imgIds)))
        
        cv2.imshow("pos", image_debug)
        if cv2.waitKey(0):
            pass
        '''
        if cv2.waitKey() & 0xFF == ord('q'):
            break
        '''
    with open(model_name + '_results.json', 'w') as f:
        json.dump(results, f)

    cocoDt = cocoGt.loadRes(model_name + '_results.json')
    cocoEval = pycocotools.cocoeval.COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.params.imgIds = imgIds
    
    #cocoEval.params.catIds = [1]
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def main():

    args = parse_args()

    if not os.path.isfile(args.model):
        raise SystemExit('ERROR: file (%s) not found!' % args.model)

    model_name = args.model.split(".")[0]
    model_name = model_name.split("/")[-1]

    input_dim = model_name.split('_')[-1]
    if 'x' in input_dim:
        dim_split = input_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad input_dim (%s)!' % input_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else :
        print("W, H parsing error")


    trt_pose = TrtPOSE(args.model, (h, w))

    loop_and_detect(model_name, args.video, trt_pose, (h, w), conf_th=0.9)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
