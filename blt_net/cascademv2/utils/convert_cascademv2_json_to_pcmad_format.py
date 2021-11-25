import pickle
import json
from blt_net.eval_city.eval_script.eval_demo import get_COCO_gt

input_filepath = '../../data/output/inference/Citypersons/raw_inference/all_results.json'
output_filename = "../../data/input/PCMAD/Citypersons/cascademv2.pkl"

with open(input_filepath) as f:
    proposals = json.load(f)

coco_gt = get_COCO_gt('./../../eval_city/val_gt.json')

uncovered_gt_counter = 0
uncovered_gt_height_arr = []
valid_gt_counter = 0

all_props = {}
for index in range(len(coco_gt.imgs)):
    image_id = index + 1
    img_name = coco_gt.imgs[image_id]['im_name']
    
    img_proposals = []
    for prop in proposals:
        
        if prop['image_id'] == image_id:
            bbox = prop['bbox']
            confidence = prop['confidence']
            bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[1] + bbox[3], confidence]
            
            img_proposals.append(bbox)
        
        all_props[img_name] = img_proposals

with open(output_filename, 'wb') as fid:
    pickle.dump(all_props, fid)
