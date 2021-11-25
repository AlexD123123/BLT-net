from __future__ import division
import os
import numpy as np
from blt_net.cascademv2.config_cascademv2 import get_cfg_defaults
import argparse
from blt_net.cascademv2.utils.general_utils import create_logger
import sys


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


def main(argv):
    
    parser = argparse.ArgumentParser(description='Train config.')
    parser.add_argument('--cfg_path',
                        type=str,
                        required=False,
                        help='Path to YAML config file.')

    args = parser.parse_args()
    
    cascademv2_config = get_cfg_defaults()
    
    if args.cfg_path is None:
        cascademv2_config.merge_from_file("./trainconfigs/citypersons_config.yaml")

    else:
        cascademv2_config.merge_from_file(args.cfg_path)
    
    if not os.path.exists(cascademv2_config.out_path):
        os.makedirs(cascademv2_config.out_path, exist_ok=True)
        
    logger = create_logger(cascademv2_config.out_path)
    
    # Continue retraining from a specific epoch (add_epoch) or from the last existing epoch in the output directory (add_epoch==-1)
    if cascademv2_config.add_epoch > 0 or cascademv2_config.add_epoch == -1:
        
        max_epoch = 0
        max_model_name = ''
        # Get last model in output directory
        model_name_arr = [model_name for model_name in np.sort(os.listdir(cascademv2_config.out_path)) if model_name.endswith(('.hdf5'))]
        model_path = cascademv2_config.out_path
        
        for model_index in range(0, len(model_name_arr)):
            model_name = model_name_arr[model_index]
            
            epoch_str = model_name.split(cascademv2_config.backbone + "_e")
            
            # Skip teacher model.
            if len(epoch_str)==2:
                epoch_str = epoch_str[1]
            else:
                continue
                
            epoch_num = int(epoch_str.split("_")[0])
            
            # If we are looking for the latest epoch.
            if cascademv2_config.add_epoch == -1:
                if epoch_num > max_epoch:
                    max_epoch = epoch_num
                    max_model_name = model_name
            else:
                # If we are looking for a specific model.
                if epoch_num == cascademv2_config.add_epoch:
                    max_epoch = epoch_num
                    max_model_name = model_name
        
        # If we found a model.
        if len(max_model_name) > 0:
            weight_path = os.path.join(cascademv2_config.out_path, max_model_name)
            cascademv2_config.add_epoch = max_epoch
        else:
            weight_path = cascademv2_config.initial_model
            cascademv2_config.add_epoch = 0
    else:
        # If epoch == 0, start from initial model
        # Use configuration definition
        weight_path = cascademv2_config.initial_model
        
    # Define the CascadeMV2 network and start training.
    
    if cascademv2_config.steps == 2:
        from blt_net.cascademv2.core.modes.model_2step import Model_2step
        model = Model_2step()

    else:
        raise NotImplementedError('Not implemented {} steps'.format(cascademv2_config.steps))
    
    # Train model on image sizes defined as in random_crop
    cascademv2_config.img_input = cascademv2_config.random_crop
    
    model.initialize(cascademv2_config, logger)
    model.creat_model(cascademv2_config, phase='train')
    model.set_data_loader(cascademv2_config, train_filename=cascademv2_config.train_filename, val_filename=cascademv2_config.val_filename, phase='train')
    model.train_model(cascademv2_config, weight_path)
    
    
    # Rvaluate model from a certian epoch ---------------------------------
    
    #if we want to run evaluation
    if cascademv2_config.eval_min_epoch>0:
        from blt_net.cascademv2.eval_cascademv2 import eval_on_batch
        eval_on_batch(cascademv2_config)
  
  
if __name__ == '__main__':
    main(sys.argv[1:])