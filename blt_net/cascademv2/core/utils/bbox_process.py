from __future__ import division
import numpy as np
# import matplotlib.pyplot as plt
from blt_net.cascademv2.core.utils.bbox_transform  import bbox_transform_inv, clip_boxes, bbox_transform
from blt_net.cascademv2.core.utils import data_generators
from blt_net.cascademv2.core.utils.nms_wrapper import nms


def format_img(img, C):
    """ formats the image channels based on config """
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]

    img = np.expand_dims(img, axis=0)
    return img


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def filter_negboxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws <= min_size) | (hs <= min_size))[0]
    return keep


def compute_targets(ex_rois, gt_rois, classifier_regr_std,std):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    # Optionally normalize targets by a precomputed mean and stdev
    if std:
        targets = targets/np.array(classifier_regr_std)
    return targets


# for training model_2step and model_3step
# 	get all regressed bboxes
# 	remove from them boxes that overlap the ignore regions

def get_target_1st(all_anchors, regr_layer, img_data, C, roi_stride=10, igthre=0.5,posthre=0.7,negthre=0.5, input_image=None):

    A = np.copy(all_anchors[:,:4])
    y_cls_batch, y_regr_batch = [], []

    # for each image in batch
    for i in range(regr_layer.shape[0]):

        # ground truth anchor
        gta = np.copy(img_data[i]['bboxes'])
        num_bboxes = len(gta)

        ignoreareas = img_data[i]['ignoreareas']
        proposals = np.ones_like(all_anchors)

        # get the regressed deltas with respect to the anchors and from them calculate the aboslute proposals
        bbox_deltas = regr_layer[i, :, :]
        bbox_deltas = bbox_deltas * np.array(C.classifier_regr_std).astype(dtype=np.float32)
        
        # get proposals using the anchors (A) and the predicted distance from them (bbox_deltas)
        proposals[:, : 4] = bbox_transform_inv(A, bbox_deltas)
        proposals = clip_boxes(proposals, [C.random_crop[0], C.random_crop[1]])

        y_alf_cls,  y_alf_regr = data_generators.calc_targets(C, num_bboxes, gta, ignoreareas, proposals, igthre=igthre, posthre=posthre, negthre=negthre, roi_stride=roi_stride)

        y_cls_batch.append(y_alf_cls)
        y_regr_batch.append(y_alf_regr)
        
    y_cls_batch = np.concatenate(np.array(y_cls_batch), axis=0)
    y_regr_batch = np.concatenate(np.array(y_regr_batch), axis=0)

    return [y_cls_batch, y_regr_batch]


# Called during training of 3step model
def get_target_2nd(anchors, regr_layer, img_data, C, roi_stride=10,igthre=0.5,posthre=0.7,negthre=0.5):
    y_cls_batch, y_regr_batch = [], []
    anc_len = anchors.shape[1]
    for i in range(regr_layer.shape[0]):
        gta = img_data[i]['bboxes']
        num_bboxes = len(gta)
        ignoreareas = img_data[i]['ignoreareas']
        proposals = np.copy(anchors[i,:,:])

        bbox_deltas = regr_layer[i,:,:]

        bbox_deltas = bbox_deltas * np.array(C.classifier_regr_std).astype(dtype=np.float32)
        proposals[:,:4] = bbox_transform_inv(proposals[:,:4], bbox_deltas)
        proposals = clip_boxes(proposals, [C.random_crop[0], C.random_crop[1]])
        
        keep = filter_negboxes(proposals, roi_stride)
        proposals[keep, -1] = 0
        
        y_alf_cls, y_alf_regr = data_generators.calc_targets(C, num_bboxes, gta, ignoreareas, proposals, igthre=igthre, posthre=posthre, negthre=negthre, debug=False)
        
        y_cls_batch.append(y_alf_cls)
        y_regr_batch.append(y_alf_regr)
        
    y_cls_batch = np.concatenate(np.array(y_cls_batch), axis=0)
    y_regr_batch = np.concatenate(np.array(y_regr_batch), axis=0)

    return [y_cls_batch, y_regr_batch]


# Called during training the 3step model
def generate_pp_2nd(all_anchors, regr_layer, C):
    
    A = np.copy(all_anchors[:, : 4])
    proposals_batch = []
    for i in range(regr_layer.shape[0]):
        proposals = np.ones_like(all_anchors)
        bbox_deltas = regr_layer[i,:,:]
        bbox_deltas = bbox_deltas * np.array(C.classifier_regr_std).astype(dtype=np.float32)
        proposals[:,:4] = bbox_transform_inv(A, bbox_deltas)
        proposals = clip_boxes(proposals, [C.random_crop[0], C.random_crop[1]])
        proposals_batch.append(np.expand_dims(proposals,axis=0))
        
    return np.concatenate(proposals_batch, axis=0)


# Called during inference of the 2stage and 3stage model
def pred_pp_1st(anchors, cls_pred, regr_pred, C):
    A = np.copy(anchors[:, :4])
    scores = cls_pred[0, :, :]
    bbox_deltas = regr_pred.reshape((-1, 4))
    bbox_deltas = bbox_deltas * np.array(C.classifier_regr_std).astype(dtype=np.float32)

    # get proposals using the anchors (A) and the predicted distance from them (bbox_deltas)
    proposals = bbox_transform_inv(A, bbox_deltas)
    proposals = clip_boxes(proposals, [C.img_input[0], C.img_input[1]])
    proposals = np.concatenate((proposals, scores), axis=-1)
    return proposals


# Called during inference of the 3stage model
def pred_pp_2nd(anchors, cls_pred, regr_pred, C):
    scores = cls_pred[0, :, :]
    bbox_deltas = regr_pred.reshape((-1, 4))
    bbox_deltas = bbox_deltas * np.array(C.classifier_regr_std).astype(dtype=np.float32)

    # get proposals using the anchors (A) and the predicted distance from them (bbox_deltas)
    anchors[:, :4] = bbox_transform_inv(anchors[:, :4], bbox_deltas)
    anchors[:, :4] = clip_boxes(anchors[:, :4], [C.random_crop[0], C.random_crop[1]])
    
    proposals = np.concatenate((anchors, scores), axis=-1)
    return proposals


# Used during inference of all models (regression from relative offset to absolute coordinates)
def pred_det(anchors, cls_pred, regr_pred, C, step=1, anchors_l1=[], dt_min_anchor_width=0, dt_max_anchor_width=10000):
    if step == 1:
        scores = cls_pred[0, :, :]
        
    elif step == 2:
        scores = cls_pred[0, :, :]
        #scores = anchors[:, -1:] * cls_pred[0, :, :]
        
    elif step == 3:
        scores = anchors[:, -2:-1] * anchors[:, -1:] * cls_pred[0, :, :]
        
    A = np.copy(anchors[:, :4])
    bbox_deltas = regr_pred.reshape((-1, 4))
    bbox_deltas = bbox_deltas * np.array(C.classifier_regr_std).astype(dtype=np.float32)

    #get proposals using the anchors (A) and the predicted distance from them (bbox_deltas)
    proposals = bbox_transform_inv(A, bbox_deltas)
    proposals = clip_boxes(proposals, [C.img_input[0], C.img_input[1]])
    
    anchors_l1_width = []
    
    # Filter by minimum anchor width of the original anchors
    if len(anchors_l1) > 0:
        anchors_l1_width = anchors_l1[:, 2] - anchors_l1[:, 0]
        anchors_l1_width = np.reshape(anchors_l1_width, [len(anchors_l1_width), 1])

        keep = np.where(np.logical_and(anchors_l1_width>=dt_min_anchor_width, anchors_l1_width<=dt_max_anchor_width))[0]
        
        anchors_l1 = anchors_l1[keep, :]
        anchors_width = anchors_l1_width[keep]
        proposals = proposals[keep, :]
        scores = scores[keep]
        anchors = anchors[keep, :]

    # Filter bboxes that are smaller than a certain width and height (as defined by C.roi_stride)
    keep = filter_boxes(proposals, C.roi_stride)
    proposals = proposals[keep, :]
    scores = scores[keep]
    if len(anchors_l1) > 0:
        anchors_l1 = anchors_l1[keep, :]
    anchors = anchors[keep, :]

    order = scores.ravel().argsort()[::-1]
    order = order[:C.pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]
    if len(anchors_l1) > 0:
        anchors_l1 = anchors_l1[order, :]
    anchors = anchors[order, :]
    
    keep = np.where(scores > C.scorethre)[0]
    proposals = proposals[keep, :]
    scores = scores[keep]
    if len(anchors_l1) > 0:
        anchors_l1 = anchors_l1[keep, :]
    anchors = anchors[keep, :]
    
    proposals_pre_nms = proposals
    scores_pre_nms = scores
    
    p = np.hstack((proposals, scores))
    p = np.array(p, dtype='f')
    
    keep = nms(p, C.overlap_thresh, usegpu=True, gpu_id=0)
    #keep = nms(p, C.overlap_thresh, usegpu=False)
    
    keep = keep[:C.post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]
    if len(anchors_l1) > 0:
        anchors_l1 = anchors_l1[keep, :]
    anchors = anchors[keep, :]
    
    return proposals, scores, proposals_pre_nms, scores_pre_nms, anchors[:, 0:4], anchors_l1[:, 0:4]