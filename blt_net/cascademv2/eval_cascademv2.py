from __future__ import division
import os
import pickle
import numpy as np
import blt_net.cascademv2.utils.benchmark_utils as benchmark_utils
import ntpath
import cv2
from blt_net.cascademv2.utils.general_utils import create_logger
import sys
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # set to 0 when only 1 GPU is available and 1 when 2 GPUs are available
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


def evaluate_model(cascademv2_config, model_path, model_name):
    
    eval_cascademv2_config = cascademv2_config.clone()
    
    model_path = eval_cascademv2_config.model_name
    
    if not os.path.isfile(model_path):
        print('Model {} does not exist. Skipping...'.format(model_path))
        return
    
    eval_cascademv2_config.img_input = eval_cascademv2_config.eval_im_input
    
    out_path = os.path.join(eval_cascademv2_config.eval_output_root, 'raw_inference')
    
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
        
    eval_cascademv2_config.out_path = out_path

    os.makedirs(out_path, exist_ok=True)
    logger = create_logger(eval_cascademv2_config.out_path, show_in_stdout=False)
    
    print('Evaluating model {}'.format(model_path))
    # ---------------------------------------------------------
    
    with open(eval_cascademv2_config.eval_filename, 'rb') as fid:
        val_data = pickle.load(fid)
    
    # ---------------------------------------------------------
    
    if cascademv2_config.steps == 2:
        from blt_net.cascademv2.core.modes.model_2step import Model_2step
        model = Model_2step()
    
    else:
        raise NotImplementedError('Not implement {} or more steps'.format(cascademv2_config.steps))
    
    model.initialize(eval_cascademv2_config, logger)
    
    model.creat_model(eval_cascademv2_config, phase='inference')
    model.load_model(model_path)
    
    eval_model_on_dataset(model, eval_cascademv2_config, val_data)
    
    benchmark_utils.convert_results(out_path, eval_cascademv2_config.eval_dataset_type)


def eval_model_on_dataset(model, cascademv2_config, val_data):
    
    filenames = []
    for f_index in range(len(val_data)):
        filenames.append(val_data[f_index]['filepath'])
    
    ind = np.argsort(filenames)
    for f_index in range(len(ind)):
        # for f_index in range(0,10):
        f_ind = ind[f_index]
        filepath = val_data[f_ind]['filepath']
        
        gt_arr = []
        if 'bboxes' in val_data[f_ind].keys():
            gt_arr = val_data[f_ind]['bboxes']
        
        head, file_name = ntpath.split(filepath)
        
        
        file_name = os.path.splitext(file_name)
        file_name = file_name[0]
        res_txt_filename = os.path.join(model.out_path, file_name + '.txt')
        res_img_filename = ''
        
        if cascademv2_config.create_image:
            res_img_filename = os.path.join(model.out_path, file_name + '.png')
        
        img = cv2.imread(filepath)
        
        model.test_model(cascademv2_config, img, gt_arr=gt_arr, res_txt_filename=res_txt_filename, res_img_filename=res_img_filename)
        
        
def main(argv):
    
    import argparse
    
    from blt_net.cascademv2.config_cascademv2 import get_cfg_defaults
    
    cascademv2_config = get_cfg_defaults()
    
    parser = argparse.ArgumentParser(description='eval config.')
    parser.add_argument('--cfg_path',
                        type=str,
                        required=False,
                        help='Path to YAML config file.')
    
    args = parser.parse_args()
    if args.cfg_path is None:
        cascademv2_config.merge_from_file("./trainconfigs/citypersons_config.yaml")
    else:
        cascademv2_config.merge_from_file(args.cfg_path)
  
    evaluate_model(cascademv2_config, cascademv2_config.out_path, cascademv2_config.model_name)
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
