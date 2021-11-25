# This module provides utilities for the merge package.
import numpy as np

# Calculate the bounding box of two boxes.
def calc_bounding_box(bbox1, bbox2):
	x_start1, y_start1, x_end1, y_end1 = bbox1
	x_start2, y_start2, x_end2, y_end2 = bbox2
	
	x_left = min(x_start1, x_start2)
	y_top = min(y_start1, y_start2)
	x_right = max(x_end1, x_end2)
	y_bottom = max(y_end1, y_end2)
	
	bbox = [x_left, y_top, x_right, y_bottom]
	bbox_area = (y_bottom - y_top) * (x_right - x_left)
	
	return np.array(bbox), bbox_area


# Calculate the coverage of crops with respect to image, considering the scaling factor of crops.
def calc_coverage(img_size, bbox_arr, weight_arr):
	
	img = np.zeros(img_size)
	img_scaled_ = np.zeros((len(bbox_arr), img_size[0], img_size[1]))
	
	total_pxls_processed_with_resizing = 0
	total_pxls_processed_no_resizing = 0
	
	bbox_arr = np.array(bbox_arr)
	for bbox_index, bbox in enumerate(bbox_arr):
		bbox = bbox.astype('int')
		x_start, y_start, x_end, y_end = bbox
		height = y_end - y_start
		width = x_end - x_start
		
		resize_weight = weight_arr[bbox_index]
		resized_height = height*resize_weight
		resized_width = width*resize_weight
		
		total_pxls_processed_no_resizing = total_pxls_processed_no_resizing + height * width
		total_pxls_processed_with_resizing = total_pxls_processed_with_resizing + resized_height * resized_width
		
		img[y_start:(y_end + 1), x_start:(x_end + 1)] = 1
		img_scaled_[bbox_index, y_start:(y_end + 1), x_start:(x_end + 1)] = resize_weight
	
	if total_pxls_processed_no_resizing < total_pxls_processed_with_resizing:
		print('Total resized pixels is bigger than without resizing')
		
	if len(bbox_arr) == 0:
		img_scaled_ = np.zeros(img_size)
	img_scaled = np.max(img_scaled_, axis=0)
	
	unique_processed_pixels_no_resizing = np.sum(img)
	unique_processed_pixels_with_resizing = np.sum(img_scaled)

	return total_pxls_processed_no_resizing, total_pxls_processed_with_resizing, unique_processed_pixels_no_resizing, unique_processed_pixels_with_resizing