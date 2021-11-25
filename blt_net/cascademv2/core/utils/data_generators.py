from __future__ import absolute_import
from __future__ import division
import random
import numpy as np
from blt_net.cascademv2.core.utils import data_augment

from blt_net.cascademv2.core.utils.cython_bbox import bbox_overlaps
from blt_net.cascademv2.core.utils.bbox_process import  compute_targets
from blt_net.cascademv2.core.utils.bbox import box_op


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _ratio_enum2(anchor, ratios):
    w, h, x_ctr, y_ctr = _whctrs(anchor)

    ws = w/np.ones((len(ratios)))
    hs = np.round(ws / ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def get_anchors(img_width, img_height, feat_map_sizes, anchor_box_scales, anchor_ratios):
    
    downscale = np.asarray([[8],[16],[32],[64]])
    ancs = []
    num_anchors = np.zeros((len(downscale)),dtype=np.int)
    
    for layer in range(len(downscale)):
        
        anchor_scales = anchor_box_scales[layer] / downscale[layer]
        base_anchor = np.array([1, 1, downscale[layer], downscale[layer]]) - 1
        ratio_anchors = _ratio_enum2(base_anchor, anchor_ratios[layer])
        anchors = np.vstack([_scale_enum(ratio_anchors[i, :], anchor_scales)
                             for i in range(ratio_anchors.shape[0])])
        num_anchors[layer] = len(anchors)

        output_width, output_height = feat_map_sizes[layer][1], feat_map_sizes[layer][0]

        shift_x = np.arange(output_width) * downscale[layer]
        shift_y = np.arange(output_height) * downscale[layer]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        all_anchors = np.expand_dims(anchors, axis=0) + np.expand_dims(shifts, axis=0).transpose((1, 0, 2))
        all_anchors = np.reshape(all_anchors, (-1, 4))
        # only keep anchors inside the image
        all_anchors[:, 0][all_anchors[:, 0] < 0] = 0
        all_anchors[:, 1][all_anchors[:, 1] < 0] = 0
        all_anchors[:, 2][all_anchors[:, 2] >= img_width] = img_width - 1
        all_anchors[:, 3][all_anchors[:, 3] >= img_height] = img_height - 1
        all_anchors = np.concatenate((all_anchors, np.ones((all_anchors.shape[0], 1))), axis=-1)
        ancs.append(all_anchors)
        
    return np.concatenate(ancs, axis=0), num_anchors


def calc_target(C, num_bboxes, gta, ignoreareas, proposals, igthre=0.5, posthre=0.5, negthre=0.3, img=[], roi_stride=0):
    
    proposals = proposals.copy()
    
    # calculate the valid anchors (without those in the ignore areas and outside the image)
    if len(ignoreareas) > 0:
        ignore_overlap = box_op(np.ascontiguousarray(proposals[:, :4], dtype=np.float),
                                np.ascontiguousarray(ignoreareas, dtype=np.float))
        ignore_sum = np.sum(ignore_overlap, axis=1)
        proposals[ignore_sum > igthre, -1] = 0
        
        #debugging code
        ignored_proposal_index_arr = np.where(ignore_sum > igthre)[0]
        ignored_anchors = proposals[ignored_proposal_index_arr, :4]
        # utils.show_anchors(input_image[i].copy(), (ignored_anchors, gta), 'level 2: positive anchors vs. gt', color_arr=((0, 255, 255), (255, 255, 0)))
    
    from blt_net.cascademv2.core.utils.bbox_process import filter_negboxes
    dont_keep = filter_negboxes(proposals, roi_stride)
    proposals[dont_keep, -1] = 0
    
    # get the indexes of the anchors that are not overlapping the ignore areas
    valid_idxs = np.where(proposals[:, -1] == 1)[0]
    
    # initialise empty output objectives
    y_alf_overlap = np.zeros((proposals.shape[0], 1))
    y_alf_negindex = np.zeros((proposals.shape[0], 1))
    y_is_box_valid = np.zeros((proposals.shape[0], 1))
    y_alf_regr = np.zeros((proposals.shape[0], 4))
    
    # remove anchors that are overlapping the ignore regions
    valid_anchors = proposals[valid_idxs, :]
    valid_alf_overlap = np.zeros((valid_anchors.shape[0], 1))
    valid_is_box_valid = np.zeros((valid_anchors.shape[0], 1))
    valid_alf_regr = np.zeros((valid_anchors.shape[0], 4))
    
    # utils.show_anchors(img, (all_anchors, gta), 'all anchors vs. gt', color_arr=((0, 255, 255), (255, 255, 0)))
    
    # get for training only anchors that overlap the ground truth, the rest is defined as negative (below a IoU threshold)
    # if there are any GTs
    if num_bboxes > 0:
        valid_overlap = bbox_overlaps(np.ascontiguousarray(valid_anchors, dtype=np.float),
                                      np.ascontiguousarray(gta, dtype=np.float))
        
        # find for every anchor to which gt box is the closest
        argmax_overlaps = valid_overlap.argmax(axis=1)
        # for all valid anchors get the IOU of the anchors maximum overlap with GT
        max_overlaps = valid_overlap[np.arange(len(valid_idxs)), argmax_overlaps]
        
        # for each GT get the anchor index that maximum overlaps it
        gt_argmax_overlaps = valid_overlap.argmax(axis=0)
        # for each GT get the IOU of the anchor that maximum overlaps it
        gt_max_overlaps = valid_overlap[gt_argmax_overlaps, np.arange(num_bboxes)]
        
        # for each GT mark the anchor that maximally overlaps it (regardless of its IOU) as positive
        gt_argmax_overlaps = np.where(valid_overlap == gt_max_overlaps)[0]
        valid_alf_overlap[gt_argmax_overlaps] = 1
        
        # get only the valid anchors that are above a specific IOU given threshold
        valid_alf_overlap[max_overlaps >= posthre] = 1
        
        # in any case get the three most overlapping anchors to each GT
        for i in range(len(gta)):
            inds = valid_overlap[:, i].ravel().argsort()[-3:]
            valid_alf_overlap[inds] = 1
        
        # get positives labels
        fg_inds = np.where(valid_alf_overlap == 1)[0]
        valid_is_box_valid[fg_inds] = 1
        
        anchor_box = valid_anchors[fg_inds, :4]
        gt_box = gta[argmax_overlaps[fg_inds], :]

        # if len(img)>0:
        # 	show_anchors(img.copy(), (anchor_box.copy(), gta.copy()), 'level 1: pos. anchors vs. gt', color_arr=((0, 255, 255), (255, 255, 0)))

        # calculate regression values to GTs only for positive anchors
        valid_alf_regr[fg_inds, :] = compute_targets(anchor_box, gt_box, C.classifier_regr_std, std=True)
        bg_inds = np.where((max_overlaps < negthre) & (valid_is_box_valid.reshape((-1)) == 0))[0]
        valid_is_box_valid[bg_inds] = 1
        
        # transform to the original overlap and validbox
        y_alf_overlap[valid_idxs, :] = valid_alf_overlap
        y_is_box_valid[valid_idxs, :] = valid_is_box_valid
        y_alf_regr[valid_idxs, :] = valid_alf_regr
        y_alf_negindex = y_is_box_valid - y_alf_overlap

    # else: #(TODO check if this is needed)
    # 	y_alf_negindex[valid_idxs, :] = np.ones((valid_idxs.shape[0], 1))
    
    # set the y[0] to contain positive examples and y[1] to contain negative examples
    y_alf_cls = np.expand_dims(np.concatenate([y_alf_overlap, y_alf_negindex], axis=1), axis=0)
    y_alf_regr = np.expand_dims(np.concatenate([y_alf_overlap, y_alf_regr], axis=1), axis=0)
    
    return y_alf_cls, y_alf_regr


def calc_targets(C, num_bboxes, gta, ignoreareas, proposals, igthre=0.5, posthre=0.5, negthre=0.3, img=[], roi_stride=0):
    
    y_alf_cls, y_alf_regr = calc_target(C, num_bboxes, gta, ignoreareas, proposals,
                                        igthre=igthre, posthre=posthre, negthre=negthre, img=img, roi_stride=roi_stride)
    
    return y_alf_cls, y_alf_regr
  

def calc_target_multilayer(C, img_data, anchors, igthre=0.5, posthre=0.5, negthre=0.3, img=[]):

    # by default all anchors are valid (last column)
    all_anchors = np.copy(anchors)
    
    num_bboxes = len(img_data['bboxes'])
    gta = np.copy(img_data['bboxes'])
    ignoreareas = img_data['ignoreareas']
    
    return calc_targets(C, num_bboxes, gta, ignoreareas, all_anchors, igthre=igthre, posthre=posthre, negthre=negthre, img=img)

    
# Generating the training data
def get_target(anchors, all_img_data, C, batchsize=4, net='2step', igthre=0.5,posthre=0.5, negthre=0.3):

    random.shuffle(all_img_data)
    
    current = 0
    
    while True:
        
        x_img_batch, y_cls_batch, y_regr_batch, img_data_batch, orig_img_batch = [], [], [], [], []

        if current >= len(all_img_data)-batchsize:
            
            random.shuffle(all_img_data)
            current = 0

        for img_data in all_img_data[current:current+batchsize]:
        #for img_data in all_img_data[0:1]:
            
            # augment1: brightness
            # augment2: horizontal flip, coordinates
            
            img_data, x_img = data_augment.augment(img_data, C, augment_brightness=C.augmentBrightness, augment_crop=C.augmentCrop)
            
            orig_img_batch.append(x_img)

            y_cls, y_regr = calc_target_multilayer(C, img_data, anchors, igthre=igthre, posthre=posthre, negthre=negthre, img=x_img)
    
            x_img = x_img.astype(np.float32)
            x_img[:, :, 0] -= C.img_channel_mean[0]
            x_img[:, :, 1] -= C.img_channel_mean[1]
            x_img[:, :, 2] -= C.img_channel_mean[2]
            x_img = np.expand_dims(x_img, axis=0)

            x_img_batch.append(x_img)
            y_cls_batch.append(y_cls)
            y_regr_batch.append(y_regr)
            img_data_batch.append(img_data)
            
        x_img_batch = np.concatenate(np.array(x_img_batch),axis=0)
        y_cls_batch = np.concatenate(np.array(y_cls_batch), axis=0)
        y_regr_batch = np.concatenate(np.array(y_regr_batch), axis=0)
        
        current += batchsize
        
        if net == '2step':
            yield np.copy(x_img_batch), [np.copy(y_cls_batch), np.copy(y_regr_batch)], np.copy(img_data_batch), np.copy(orig_img_batch)
        else:
            yield np.copy(x_img_batch), [np.copy(y_cls_batch), np.copy(y_regr_batch)]


