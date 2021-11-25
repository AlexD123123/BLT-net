import numpy as np
import cv2
import yaml

def createImage(image, bbox_arr, score_arr, save_image_filename, gt_bbox_arr=None, proposals_pre_nms=None, scores_pre_nms=None,
                anchors=[], anchors_l1=[],
                show_image=False, img_title='', det_extra_data=[], reasonable_gt_height=50):
    
    image = image.copy()
    all_bbox_arr = []
    all_scores_arr = []
    color_arr = []
    extra_data_arr = []
    
    # pre-NMS detections
    if proposals_pre_nms is not None and scores_pre_nms is not None:
        all_bbox_arr.append(proposals_pre_nms.copy())
        all_scores_arr.append(scores_pre_nms.copy())
        color_arr.append((0, 255, 255)) #yellow
        extra_data_arr.append([])
    
    #Ground truth
    if gt_bbox_arr is not None:
        
        reasonable_gt_arr = []
        unreasonable_gt_arr = []
        for gt_bbox in gt_bbox_arr:
            gt_height = gt_bbox[3]
            if gt_height >= reasonable_gt_height:
                reasonable_gt_arr.append(gt_bbox)
            else:
                unreasonable_gt_arr.append(gt_bbox)
        
        if len(reasonable_gt_arr)>0:
            all_bbox_arr.append(reasonable_gt_arr)
            gt_score_arr = len(reasonable_gt_arr) * [-1]
            all_scores_arr.append(gt_score_arr)
            color_arr.append((0, 0, 255)) #red
            extra_data_arr.append([])
        
        if len(unreasonable_gt_arr)>0:
            all_bbox_arr.append(unreasonable_gt_arr)
            gt_score_arr = len(unreasonable_gt_arr) * [-1]
            all_scores_arr.append(gt_score_arr)
            color_arr.append((255, 0, 255)) #magenta
            extra_data_arr.append([])
    
    # anchors step1
    if anchors_l1 is not None:
        all_bbox_arr.append(anchors_l1.copy())
        anchors_l1_score_arr = len(anchors_l1) * [-1]
        all_scores_arr.append(anchors_l1_score_arr)
        color_arr.append((180, 0, 180)) #dark magenta
        extra_data_arr.append([])
    
    # anchors higher level
    if anchors is not None:
        all_bbox_arr.append(anchors.copy())
        anchors_score_arr = len(anchors) * [-1]
        all_scores_arr.append(anchors_score_arr)
        color_arr.append((255, 255, 255)) #white
        extra_data_arr.append([])
    
    # Detections
    all_bbox_arr.append(bbox_arr.copy())
    all_scores_arr.append(score_arr.copy())
    color_arr.append((255, 255, 0))  # cyan
    extra_data_arr.append(det_extra_data)
    
    # For each type of boxes
    for k in range(len(all_bbox_arr)):
        
        bbox_arr = all_bbox_arr[k]
        score_arr = all_scores_arr[k]
        box_color = color_arr[k]
        extra_data = extra_data_arr[k]
    
        for i, bbox in enumerate(bbox_arr):
            
            bbox = bbox.astype(int)
            
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            score = score_arr[i]
            
            extra_str = ''
            if len(extra_data)>0:
                if len(extra_data)>i:
                    extra_str = extra_data[i]
            
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=box_color, thickness=1)
            
            score_txt = ''
            if score>=0:
                score_txt = np.round(score, 2)
            
            cv2.putText(image, "{} {}".format(score_txt, extra_str), (bbox[0], bbox[1]), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)
            
    if len(save_image_filename)>0:
        cv2.imwrite(save_image_filename, image)
    
    if show_image:
        #image = cv2.resize(image, (1280, 640), interpolation=cv2.INTER_LANCZOS4)
        cv2.imshow(img_title, image)
        cv2.waitKey(0)
        
    return image


def show_anchors(im, anchors_arr, win_name, color_arr=None):
    im = im.copy()
    if color_arr is None:
        import matplotlib.cm as cm
        color_arr = cm.rainbow(np.linspace(0, 1, len(anchors_arr)))
        color_arr = np.floor(color_arr[:, 0:3] * 255)
        color_arr = color_arr.astype(np.int)
    
    for ind1 in range(len(anchors_arr)):
        
        all_bbox_arr = anchors_arr[ind1]
        
        box_color = color_arr[ind1]
        
        for s in range(len(all_bbox_arr)):
            bbox_arr = np.floor(all_bbox_arr[s])
            bbox_arr = bbox_arr.astype(int)
            
            cv2.rectangle(im, (bbox_arr[0], bbox_arr[1]), (bbox_arr[2], bbox_arr[3]), color=box_color, thickness=1)
    # cv2.rectangle(im, (bbox_arr[0], bbox_arr[1]), (bbox_arr[2], bbox_arr[3]), color=(255,255,0), thickness=1)
    
    cv2.imshow(win_name, im)
    
    cv2.waitKey(0)
    
 

def create_video(image_folder_path, video_filename):
    import cv2
    import os
    
    images = [img for img in sorted(os.listdir(image_folder_path)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder_path, images[0]))
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(video_filename, 0, 20, (width, height))
    
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder_path, image)))
    
    cv2.destroyAllWindows()
    video.release()


# calculate IOU between two boxes represented by (x_top_left, y_top_left, x_bottom_right, y_bottom, right)
# IOU = intersect(A, B)/(A + B - intersect(A, B)

def calc_iou(bbox1, bbox2):
    x_start1, y_start1, x_end1, y_end1 = bbox1
    x_start2, y_start2, x_end2, y_end2 = bbox2
    
    x_left = max(x_start1, x_start2)
    y_top = max(y_start1, y_start2)
    x_right = min(x_end1, x_end2)
    y_bottom = min(y_end1, y_end2)
    
    intersection_area = 0
    
    bb1_area = (x_end1 - x_start1) * (y_end1 - y_start1)
    bb2_area = (x_end2 - x_start2) * (y_end2 - y_start2)
    
    unite_area = bb1_area + bb2_area - intersection_area
    
    # if there is no overlap
    if (x_right < x_left) or (y_bottom < y_top):
        intersection_area = 0
    
    else:
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # compute the area of both AABBs
    
    iou = intersection_area / unite_area
    
    return iou, bb1_area, bb2_area, intersection_area, unite_area


def create_logger(logger_path, show_in_stdout=True):
    
    import logging
    import os
    
    os.makedirs(logger_path, exist_ok=True)
    
    #logging.basicConfig(stream=sys.stderr)
    
    logger = logging.getLogger('loceng_logger')
    #logger.propagate = False
    logger.setLevel(logging.DEBUG)
    
    handler = logging.FileHandler(os.path.join(logger_path, 'logger.txt'))
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    
    if show_in_stdout:

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
    
        ch.setFormatter(formatter)
    
        logger.addHandler(ch)

    return logger


# read from yaml to dictionary
def dict_from_yaml(cfg_path):

    with open(cfg_path) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
    return cfg_dict
