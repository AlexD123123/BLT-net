from __future__ import division
import cv2
import numpy as np
import copy

import numpy.random as npr


def _brightness(image, min=0.5, max=2.0):
    '''
    Randomly change the brightness of the input image.
    Protected against overflow.
    '''
    
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_br = np.random.uniform(min, max)
    # To protect against overflow: Calculate a mask for all pixels
    # where adjustment of the brightness would exceed the maximum
    # brightness value and set the value to the maximum at those pixels.
    mask = hsv[:, :, 2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
    hsv[:, :, 2] = v_channel
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(img_data, c, augment_brightness=True, augment_crop=False):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    img_data_aug = copy.deepcopy(img_data)
    
    img = cv2.imread(img_data_aug['filepath'])
    
    try:
        img_height, img_width = img.shape[:2]
    except:
        print('error in augment() {}'.format(img_data_aug['filepath']))
        img_data_aug['width'] = c.img_input[1]
        img_data_aug['height'] = c.img_input[0]
        return img_data_aug, img
    
    if augment_brightness or augment_crop:
        # random brightness
        if c.brightness and np.random.randint(0, 2) == 0:
            # img_orig = img.copy()
            img = _brightness(img, min=c.brightness[0], max=c.brightness[1])
    
    # cv2.imshow('', np.hstack((img_orig, img)))
    # cv2.destroyAllWindows()
    
    sel_id = -1
    ratio = 1
    if augment_crop:
        # random horizontal flip
        if c.use_horizontal_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)
            if len(img_data_aug['bboxes']) > 0:
                img_data_aug['bboxes'][:, [0, 2]] = img_width - img_data_aug['bboxes'][:, [2, 0]]
            if len(img_data_aug['ignoreareas']) > 0:
                img_data_aug['ignoreareas'][:, [0, 2]] = img_width - img_data_aug['ignoreareas'][:, [2, 0]]
        
        # use the scale factor to reduce the size of the effective crop (if scale <1 ) or stretch the size of the crop if upscale
        # the sampled crop is then resized to the original image_crop
        
        ratio = np.random.uniform(c.scale[0], c.scale[1])
        
        gts = np.copy(img_data_aug['bboxes'])
        if len(gts) > 0:
            # if there is a GT in the crop
            # select a GT and set it to be in the center of the crop, so we will make sure that at least one GT is fully contained in the crop
            sel_id = np.random.randint(0, len(gts))
        
    img_data_aug1 = copy.deepcopy(img_data_aug)
    
    img_resized, img_data_aug = get_augment(ratio, c, img.copy(), img_data_aug1, sel_id)
    
    is_debug = False
    
    if ratio>1:
        is_debug = True
        
    if is_debug:
        
        # show overlapping GT with crop
        for bbox in img_data_aug['bboxes']:
            bbox = bbox.astype(int)
            cv2.rectangle(img_resized, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=1)
        
        for bbox in img_data_aug['ignoreareas']:
            bbox = bbox.astype(int)
            cv2.rectangle(img_resized, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255, 0, 255), thickness=1)
        
        cv2.imshow('original image', img.copy())
        cv2.imshow('crop', img_resized)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    img_data_aug['width'] = c.random_crop[1]
    img_data_aug['height'] = c.random_crop[0]
    
    return img_data_aug, img_resized


def get_augment(scale_factor, c, img, img_data_aug, sel_id=-1):
    
    img_height, img_width = img.shape[:2]
    
    crop_w_h_ratio = c.random_crop[1] / c.random_crop[0]
    
    img_w_h_ratio =  img.shape[1] / img.shape[0]
    if scale_factor <= 1:
        
        if crop_w_h_ratio <= img_w_h_ratio:
            crop_h = np.asarray(scale_factor * np.asarray(img.shape[0]), dtype=np.int)
        
            crop_w = int(crop_w_h_ratio * crop_h)
        else:
            crop_w = np.asarray(scale_factor * np.asarray(img.shape[1]), dtype=np.int)
            
            crop_h = int(crop_w / crop_w_h_ratio)
        
    gts = np.copy(img_data_aug['bboxes'])
    igs = np.copy(img_data_aug['ignoreareas'])
    
    orig_gts = np.copy(img_data_aug['bboxes'])
    orig_igs = np.copy(img_data_aug['ignoreareas'])
    
    if len(gts) > 0:
        # if there is a GT in the crop
        # select a GT and set it to be in the center of the crop, so we will make sure that at least one GT is fully contained in the crop
        
        sel_center_x = int((gts[sel_id, 0] + gts[sel_id, 2]) / 2.0)
        sel_center_y = int((gts[sel_id, 1] + gts[sel_id, 3]) / 2.0)
    else:
        sel_center_x = int(np.random.randint(0, img_width - crop_w) + crop_w * 0.5)
        sel_center_y = int(np.random.randint(0, img_height - crop_h) + crop_h * 0.5)
    
    crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0))
    crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0))
    

    # if the crop size exceeds the image size, move crop to left
    if (crop_y1 + crop_h) > img.shape[0]:
        crop_y1 = img.shape[0] - crop_h - 1
    
    if (crop_x1 + crop_w) > img.shape[1]:
        crop_x1 = img.shape[1] - crop_w - 1
    
    img_crop = np.copy(img[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w])
    
    if len(igs) > 0:
        igs[:, [0, 2]] -= crop_x1
        igs[:, [1, 3]] -= crop_y1
        y_coords = igs[:, [1, 3]]
        y_coords[y_coords < 0] = 0
        y_coords[y_coords >= crop_h] = crop_h - 1
        igs[:, [1, 3]] = y_coords
        x_coords = igs[:, [0, 2]]
        x_coords[x_coords < 0] = 0
        x_coords[x_coords >= crop_w] = crop_w - 1
        igs[:, [0, 2]] = x_coords
        after_area = (igs[:, 2] - igs[:, 0]) * (igs[:, 3] - igs[:, 1])
        igs = igs[after_area > 100]
    
    if len(gts) > 0:
        before_limiting = copy.deepcopy(gts)
        gts[:, [0, 2]] -= crop_x1
        gts[:, [1, 3]] -= crop_y1
        y_coords = gts[:, [1, 3]]
        y_coords[y_coords < 0] = 0
        y_coords[y_coords >= crop_h] = crop_h - 1
        gts[:, [1, 3]] = y_coords
        x_coords = gts[:, [0, 2]]
        x_coords[x_coords < 0] = 0
        x_coords[x_coords >= crop_w] = crop_w - 1
        gts[:, [0, 2]] = x_coords
        
        before_area = (before_limiting[:, 2] - before_limiting[:, 0]) * (
                before_limiting[:, 3] - before_limiting[:, 1])
        after_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
        # gts = gts[after_area >= c.in_thre * before_area]
        gts = gts[after_area >= 16 / 0.41 * 16]
    
    img_data_aug['bboxes'] = gts
    img_data_aug['ignoreareas'] = igs

    img_crop_resized = cv2.resize(img_crop, dsize=(c.random_crop[1], c.random_crop[0]))
    
    reratioh = crop_h / c.random_crop[0]
    reratiow = crop_w / c.random_crop[1]
    if len(gts) > 0:
        gts[:, 0] = (gts[:, 0] / reratiow).astype(np.int)
        gts[:, 2] = (gts[:, 2] / reratiow).astype(np.int)
        gts[:, 1] = (gts[:, 1] / reratioh).astype(np.int)
        gts[:, 3] = (gts[:, 3] / reratioh).astype(np.int)
        w = gts[:, 2] - gts[:, 0]
        gts = gts[w >= 16, :]
    
    if len(igs) > 0:
        igs[:, 0] = (igs[:, 0] / reratioh).astype(np.int)
        igs[:, 2] = (igs[:, 2] / reratioh).astype(np.int)
        igs[:, 1] = (igs[:, 1] / reratiow).astype(np.int)
        igs[:, 3] = (igs[:, 3] / reratiow).astype(np.int)
        w, h = igs[:, 2] - igs[:, 0], igs[:, 3] - igs[:, 1]
        igs = igs[np.logical_and(w >= 8, h >= 8), :]
    
    img_data_aug['bboxes'] = gts
    img_data_aug['ignoreareas'] = igs
    
    return img_crop_resized, img_data_aug
