import numpy as np
import pickle
import matplotlib.pyplot as plt
import ntpath
import os
import cv2
import blt_net.pcmad.merge_utils as merge_utils
import time
from blt_net.pcmad.pcmad import PCMAD
import pickle

if __name__ == '__main__':
    
    dataset = 'Citypersons' #'Caltech'

    from blt_net.pcmad.config_pcmad import get_cfg_defaults
    
    pcmad_config = get_cfg_defaults()
    
    # ----------------------------------------------------------------------------------------
    #   CascadeMV2 proposals on Citypersons
    # ----------------------------------------------------------------------------------------

    proposals_filename = "./../data/input/PCMAD/Citypersons/cascademv2.pkl"

    proposals_version = 'cascademv2_citypersons'
    
    # PCMAD for Pedestron architecture
    pcmad_config.merge_from_file("./pcmad_configs/Pedestron_hrnet_cascademv2c0.3_thrs0.05.yaml")
    
    # =======================================================================================
    #results filename
    packed_proposals_filename = './../data/output/PCMAD/{}/results_{}_packed_{}.pkl'.format(dataset, proposals_version, pcmad_config.config_name)


    roi_module = PCMAD(pcmad_config.img_shape, pcmad_config.merge_grid_resolution, pcmad_config.allowed_size_arr,
                       pcmad_config.scale_factor_arr, pcmad_config.inital_padding_arr, pcmad_config.min_required_crop_padding_arr,
                       pcmad_config.proposals_min_conf, None)
    
    # Read proposals----------------------------------------------------------------------------------------
    
    with open(proposals_filename, 'rb') as fid:
        proposals_data = pickle.load(fid)
    
    # Get Validation dataset# ----------------------------------------------------------------------------------------
    if dataset =='Citypersons':
  
        val_filename = './../data/input/cascademv2/data/Citypersons/val'
        with open(val_filename, 'rb') as fid:
            val_data = pickle.load(fid)

    
    calc_statistics = True
    show_debug_image = False

    # ----------------------------------------------------------------------------------------
    
    if calc_statistics:
        initial_crops_total_processed_pixels_no_resizing_arr = []
        initial_crops_total_processed_pixels_with_resizing_arr = []
        
        total_processed_pixels_with_resizing_arr = []
        total_processed_pixels_no_resizing_arr = []
        
        unique_processed_pixels_no_resizing_arr = []
        unique_processed_pixels_with_resizing_arr = []
        
        padding_area_with_resizing_arr = []
        init_prop_area_with_resizing_arr = []
        
    # ----------------------------------------------------------------------------------------
    
    all_packed_crops_per_image_arr = {}  # all packed crops of all images
    processing_time_arr = []  # for performance profiling
    
    # store here config. params and eventually the results
    
    for f_index in range(0, len(val_data)):
        image_filepath = val_data[f_index]['filepath']
        head, file_name = ntpath.split(image_filepath)
        file_name_without_extension = file_name.split('.')[0]
        
        img = cv2.imread(image_filepath)
        
        gt_bbox_arr = val_data[f_index]['bboxes']
        
        image_filepath = os.path.join(head, file_name)
        
        print('Processing {}/{} {}..'.format(f_index, len(val_data), image_filepath))
        
        proposals = proposals_data[file_name]
        
        start_time = time.time()
        
        image_packed_crops_arr, initial_crops_arr, initial_net_crops_arr, new_net_crops_arr, initial_net_crops_per_height_arr, new_crops_per_height_arr = roi_module.pack_crops(proposals, img)
        
        temp_list = []
        for item in image_packed_crops_arr:
            temp_list.append(list(item))
        all_packed_crops_per_image_arr[file_name] = temp_list
        
        end_time = time.time()
        img_processing_time = end_time - start_time
        processing_time_arr.append(img_processing_time)
        
        # show image results
        show_debug_image = False
        if show_debug_image and len(initial_crops_arr)>0:
            
            img = cv2.imread(image_filepath)
            
            for bbox_index, bbox in enumerate(proposals):
                conf=bbox[4]
                bbox = bbox[0:4].astype('int')
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=1)
                cv2.putText(img, '{}'.format(np.round(conf,2)), (bbox[0], bbox[1]), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)
            
            for bbox_index, bbox in enumerate(initial_crops_arr):
                bbox = bbox.astype('int')
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255, 255, 255), thickness=1)
        
            for bbox_index, bbox in enumerate(image_packed_crops_arr):
                
                bbox = bbox.astype('int')
                cv2.rectangle(img, (bbox[0] - 2, bbox[1] - 2), (bbox[2] + 2, bbox[3] + 2), color=(0, 255,255), thickness=1)
         
            cv2.imshow('', img)
            cv2.waitKey()
        
        img_width = pcmad_config.img_shape[1]
        img_height = pcmad_config.img_shape[0]
        
        if calc_statistics:
            
            init_total_pxls_processed_no_resizing, init_total_pxls_processed_with_resizing, \
            init_unique_processed_pixels_no_resizing, init_unique_processed_pixels_with_resizing = merge_utils.calc_coverage(pcmad_config.img_shape, initial_crops_arr[:, 0:4], initial_crops_arr[:, 4])
            
            total_pxls_processed_no_resizing, total_pxls_processed_with_resizing, \
            unique_processed_pixels_no_resizing, unique_processed_pixels_with_resizing = merge_utils.calc_coverage(pcmad_config.img_shape, image_packed_crops_arr[:, 0:4], image_packed_crops_arr[:, 4])
            
            
            initial_crops_total_processed_pixels_no_resizing_arr.append(init_total_pxls_processed_no_resizing/ (img_height * img_width) * 100)
            initial_crops_total_processed_pixels_with_resizing_arr.append(init_total_pxls_processed_with_resizing/ (img_height * img_width) * 100)
            
            total_processed_pixels_no_resizing_arr.append(total_pxls_processed_no_resizing/ (img_height * img_width) * 100)
            total_processed_pixels_with_resizing_arr.append(total_pxls_processed_with_resizing/ (img_height * img_width) * 100)
            
            #print('Before\after: {}%\{}%'.format(total_pxls_processed_no_resizing/ (img_height * img_width) * 100, total_pxls_processed_with_resizing/ (img_height * img_width) * 100))
            if (total_pxls_processed_with_resizing/ (img_height * img_width) * 100)> 200:
                print('a')
            unique_processed_pixels_no_resizing_arr.append(unique_processed_pixels_no_resizing/ (img_height * img_width) * 100)
            unique_processed_pixels_with_resizing_arr.append(unique_processed_pixels_with_resizing/ (img_height * img_width) * 100)
            
            total_net_crop_area = 0
            for h_index, net_prop_per_height in enumerate(initial_net_crops_per_height_arr):
                
                scale_factor = pcmad_config.scale_factor_arr[h_index]
            
                for init_prop in net_prop_per_height:
                    net_area = (init_prop[3] - init_prop[1]) * (init_prop[2] - init_prop[0]) / scale_factor / scale_factor
                    total_net_crop_area += net_area
            
            if total_net_crop_area > total_pxls_processed_with_resizing:
                print('Error, not possible')
            
            init_prop_area_with_resizing_arr.append(total_net_crop_area)
            padding_area_with_resizing_arr.append(total_pxls_processed_with_resizing - total_net_crop_area)
    
    # ---------------------------------------------------------------------------
    if calc_statistics:
        init_prop_processed_pixels_with_no_resizing_arr = np.array(initial_crops_total_processed_pixels_no_resizing_arr)
        init_prop_processed_pixels_with_resizing_arr = np.array(initial_crops_total_processed_pixels_with_resizing_arr)
        
        total_processed_pixels_no_resizing_arr = np.array(total_processed_pixels_no_resizing_arr)
        total_processed_pixels_with_resizing_arr = np.array(total_processed_pixels_with_resizing_arr)
        
        unique_processed_pixels_no_resizing_arr = np.array(unique_processed_pixels_no_resizing_arr)
        unique_processed_pixels_with_resizing_arr = np.array(unique_processed_pixels_with_resizing_arr)
        
        init_net_prop_area_with_resizing_arr = np.array(init_prop_area_with_resizing_arr)
        padding_area_with_resizing_arr = np.array(padding_area_with_resizing_arr)
    
    # ---------------------------------------------------------------------------
    

    
    pack_data = {}
    pack_data['roi_config'] = pcmad_config
    pack_data['packed_crops'] = all_packed_crops_per_image_arr
    pack_data['stats'] = {}
    
    # Padded proposals before merge.
    pack_data['stats']['init_prop_processed_pixels_with_no_resizing_arr'] = init_prop_processed_pixels_with_no_resizing_arr
    pack_data['stats']['init_prop_processed_pixels_with_resizing_arr'] = init_prop_processed_pixels_with_resizing_arr
    
    # Proposals after after merging.
    pack_data['stats']['total_processed_pixels_no_resizing_arr'] = total_processed_pixels_no_resizing_arr
    pack_data['stats']['total_processed_pixels_with_resizing_arr'] = total_processed_pixels_with_resizing_arr
    
    # Proposals after after merging, unique area.
    pack_data['stats']['unique_processed_pixels_no_resizing_arr'] = unique_processed_pixels_no_resizing_arr
    pack_data['stats']['unique_processed_pixels_with_resizing_arr'] = unique_processed_pixels_with_resizing_arr
    
    # Net processed area.
    pack_data['stats']['init_net_prop_area_with_resizing_arr'] = init_net_prop_area_with_resizing_arr
    pack_data['stats']['padding_area_with_resizing_arr'] = padding_area_with_resizing_arr
    
    with open(packed_proposals_filename, 'wb') as fid:
        pickle.dump(pack_data, fid)

    data_arr =  [init_prop_processed_pixels_with_no_resizing_arr, total_processed_pixels_with_resizing_arr]
    
    title_str_arr = ['Processed area without PCMAD', 'Processed area with PCMAD']

    fig, axes = plt.subplots(1, 2)
    
    for k, data in enumerate(data_arr):
        ax = plt.subplot(2, 1, k + 1)
        ax.set_title('{}. {}: {}%+/-{}% '.format(dataset, title_str_arr[k], np.round(np.mean(data), 1), np.round(np.std(data), 1)))
        ax.hist(data, bins=60, range=[0, 200])
        ax.set_xlabel('Processed pixels %')
        
    plt.show()
    
    