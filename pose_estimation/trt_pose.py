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
        help='trt model path')

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

def loop_and_detect(model_name, video_file, trt_pose, input_shape):

    pred_boxes = None
    first_enter = 1
    fpslist = []

    vidcap = cv2.VideoCapture(video_file)
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while vidcap.isOpened():
        total_now = time.time()
        ret, image_bgr = vidcap.read()

        if not ret:
            break

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Clone 2 image for person detection and pose estimation
        
        image_per = image_rgb.copy()
        image_pose = image_rgb.copy()

        # Clone 1 image for debugging purpose
        image_debug = image_bgr.copy()

        ########### object detection box #######
        if first_enter :
            pred_boxes = [[(301,97),(369,338)]]
            first_enter = 0

        # Can not find people. Move to next frame
        if not pred_boxes:
            continue

        if True:
            for box in pred_boxes:
                cv2.rectangle(image_debug, box[0], box[1], color=(0, 255, 0),
                              thickness=3)  # Draw Rectangle with the coordinates

        # pose estimation : for multiple people
        centers = []
        scales = []
        for box in pred_boxes:
            center, scale = box_to_center_scale(box, input_shape[0], input_shape[1])
            centers.append(center)
            scales.append(scale)

        #print("pose_inference!")
        
        for center, scale in zip(centers, scales):
            trans = get_affine_transform(center, scale, 0, input_shape)
            # Crop smaller image of people
            input_img = cv2.warpAffine(
                image_pose,
                trans,
                (input_shape[0], input_shape[1]),
                flags=cv2.INTER_LINEAR)
        #cv2.imwrite("crop.png", input_img)

        output = trt_pose.estimation(model_name, input_img)

        # compute output heatmap
        coords, _ = get_final_preds(
            output,
            np.asarray(centers),
            np.asarray(scales))

        pose_preds = coords

        for coords in pose_preds:
            # Draw each point on image
            for coord in coords:
                x_coord, y_coord = int(coord[0]), int(coord[1])
                cv2.circle(image_debug, (x_coord, y_coord), 4, (255, 0, 0), 2)

        total_then = time.time()
        fps = 1/(total_then - total_now)
        fpslist.append(fps)

        text = "{:03.2f} FPS".format(fps)
        #print(text)
        cv2.putText(image_debug, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("pos", image_debug)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    print(np.mean(np.asarray(fpslist),axis=0))

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

    loop_and_detect(model_name, args.video, trt_pose, (h, w))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
