import os
import cv2
import pickle
import numpy as np
from blt_net.cascademv2.core import box_op
import json

import matplotlib.pyplot as plt

scale_size = 1
output_dir = '../../data/input/cascademv2/data/Citypersons/'
all_img_path = '/mnt/algo-datasets/DB/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/'
all_annot_path = '/mnt/algo-datasets/DB/Cityscapes/gtBboxCityPersons/'



types = ['train', 'val']
types_output = ['train', 'val']


rows, cols = 1024, 2048

#anno_path = os.path.join(all_anno_path, 'anno_'+type+'.mat')

debug_mode = False

for index, type in enumerate(types):

    cities_arr = [cityname for cityname in os.listdir(os.path.join(all_img_path, type))]
    output_type = types_output[index]
    res_path = os.path.join(output_dir, output_type)

    valid_count = 0
    iggt_count = 0
    rea_count = 0
    box_count = 0
    image_data = []

    total_images = 0
    total_images_skipped = 0
    total_images_no_boxes = 0

    for cityname in cities_arr:

        images_in_city_path = os.path.join(all_img_path, type, cityname)
        img_name_arr = [city for city in os.listdir(images_in_city_path)]

        for img_name in img_name_arr:

            img_name = os.path.splitext(img_name)
            img_name = img_name[0]
            img_name = img_name.split('_leftImg8bit')[0]

            img_path = os.path.join(all_img_path, type, cityname, '{}_leftImg8bit.png'.format(img_name))

            json_path = os.path.join(all_annot_path, type, cityname, '{}_gtBboxCityPersons.json'.format(img_name))

            if not (os.path.exists(json_path)):
                # print("{} does not exists.\n".format(json_path))
                #print("No json file: " + json_path)
                total_images_skipped = total_images_skipped + 1
                continue

            with open(json_path) as f:
                data = json.load(f)

                boxes = []
                ig_boxes = []
                vis_boxes = []

                for obj in data['objects']:
                    obj_label = obj['label']

                    if obj_label == 'ignore':
                        box = np.array([obj['bbox'][0], obj['bbox'][1], obj['bbox'][0] + obj['bbox'][2], obj['bbox'][1] + obj['bbox'][3]])
                        box = box/scale_size
                        box = box.astype(int)
                        ig_boxes.append(box)
                    else:
                        box = np.array([obj['bbox'][0], obj['bbox'][1], obj['bbox'][0] + obj['bbox'][2], obj['bbox'][1] + obj['bbox'][3]])
                        box = box / scale_size
                        box = box.astype(int)

                        vis_box = np.array([obj['bboxVis'][0], obj['bboxVis'][1], obj['bboxVis'][0] + obj['bboxVis'][2], obj['bboxVis'][1] + obj['bboxVis'][3]])
                        vis_box = vis_box / scale_size
                        vis_box = vis_box.astype(int)

                        boxes.append(box)
                        vis_boxes.append(vis_box)


                boxes = np.array(boxes)
                vis_boxes = np.array(vis_boxes)
                ig_boxes = np.array(ig_boxes)

            if debug_mode:
                img = cv2.imread(img_path)
                for i in range(len(boxes)):
                    (x1, y1, x2, y2) = boxes[i, :]*scale_size
                    # if y2-y1>50:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # y3 = int(y1+(y2-y1)/3.0)
                    # cv2.rectangle(img, (x1, y1), (x2, y3), (0, 255, 0), 2)
                    # cv2.putText(img, str(boxes[i, 0]), (x1 - 2, y1 - 2), cv2.FONT_HERSHEY_DUPLEX, 1,'blue', 1)

                for i in range(len(vis_boxes)):
                    (x1, y1, x2, y2) = vis_boxes[i, :]*scale_size
                    # if y2-y1>50:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                for i in range(len(ig_boxes)):
                    (x1, y1, x2, y2) = ig_boxes[i, :]*scale_size
                    # if y2-y1>50:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                plt.imshow(img, interpolation='bicubic')
                #print("a")

            if len(boxes) == 0:
                total_images_no_boxes = total_images_no_boxes + 1
                #print("No boxes: " + json_path)
                # continue

            valid_count += 1  # for statistical purposes

            annotation = {}
            annotation['filepath'] = img_path

            # ignore boxes that overlap the ignore and not_ignore
            if len(ig_boxes) > 0 and len(boxes) > 0:

                boxig_overlap = box_op(np.ascontiguousarray(boxes, dtype=np.float),
                                       np.ascontiguousarray(ig_boxes, dtype=np.float))

                ignore_sum = np.sum(boxig_overlap, axis=1)
                oriboxes = np.copy(boxes)
                boxes = oriboxes[ignore_sum < 0.5, :]
                vis_boxes = vis_boxes[ignore_sum < 0.5, :]

                if ignore_sum.max() >= 0.5:
                    iggt_count += len(ignore_sum) - len(boxes)
                    ig_boxes = np.concatenate([ig_boxes, oriboxes[ignore_sum >= 0.5, :]], axis=-0)
            box_count += len(boxes)

            annotation['bboxes'] = boxes
            annotation['vis_bboxes'] = vis_boxes
            annotation['ignoreareas'] = ig_boxes
            image_data.append(annotation)

            total_images = total_images + 1

    print('{} has {} images and {} valid gts'.format(type, total_images, box_count))

    with open(res_path, 'wb') as fid:
         pickle.dump(image_data, fid)
   
    