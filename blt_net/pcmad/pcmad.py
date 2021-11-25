import numpy as np
import pickle
import matplotlib.pyplot as plt
import ntpath
import os
import cv2
import blt_net.pcmad.merge_utils as merge_utils
import time

class PCMAD:
	
	def __init__(self, img_shape, merge_grid_resolution, allowed_size_arr, scale_factor_arr, inital_padding_arr, min_required_crop_padding_arr, proposals_min_conf, logger):
		
		assert ((len(img_shape) == 2) and (img_shape[0] > 0) and (img_shape[1] > 0))
		assert ((len(merge_grid_resolution) == 2) and (merge_grid_resolution[0] > 0) and (merge_grid_resolution[1] > 0))
		assert (len(allowed_size_arr) > 0)
		
		assert (len(scale_factor_arr) == len(allowed_size_arr))
		assert (len(inital_padding_arr) == len(allowed_size_arr))
		assert (len(min_required_crop_padding_arr) == len(allowed_size_arr))
		assert (proposals_min_conf >= 0)
		
		self.merge_grid_resolution = [20, 20]
		
		self.allowed_size_arr = []
		self.scale_factor_arr = []
		self.inital_padding_arr = []
		self.min_required_crop_padding_arr = []
		
		self.output_path = ''
		self.h_arr = []
		
		self.logger = logger
		self.img_shape = img_shape
		self.merge_grid_resolution = merge_grid_resolution
		self.allowed_size_arr = allowed_size_arr
		self.scale_factor_arr = scale_factor_arr
		self.inital_padding_arr = inital_padding_arr
		self.min_required_crop_padding_arr = min_required_crop_padding_arr
		self.proposals_min_conf = proposals_min_conf
		
		# update allowed size range according to the allowed_size_arr crop sizes
		self.h_arr = []
		for h_index in range(0, len(self.allowed_size_arr)):
			if h_index == 0:
				h_min = 20
			else:
				h_min = self.h_arr[h_index - 1][1]
			
			h_max = self.allowed_size_arr[h_index][-1][0]
			padding = self.min_required_crop_padding_arr[h_index][0]
			
			h_max = h_max - padding * 2
			self.h_arr.append([h_min, h_max])
	
	def calc_iou(self, bbox1, bbox2):
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
	
	# this function gets proposals and reduces the number of proposals to img_input_size[0]/grid_resolution_arr[0] * img_input_size[1]/grid_resolution_arr[1]
	# to limit the input size of the next step
	
	def preprocess_proposals(self, bbox_arr, grid_resolution_arr, img_input_size):
		
		num_rows = int(np.ceil(img_input_size[0] / grid_resolution_arr[0]))
		num_columns = int(np.ceil(img_input_size[1] / grid_resolution_arr[1]))
		
		grid = np.frompyfunc(list, 0, 1)(np.empty((num_rows, num_columns), dtype=object))
		
		coord_pairs_arr = {}
		for box_index, bbox in enumerate(bbox_arr):
			bbox_center_x = (bbox[0] + bbox[2]) / 2
			bbox_center_y = (bbox[1] + bbox[3]) / 2
			
			column_index = int(np.floor(bbox_center_x / grid_resolution_arr[0]))
			row_index = int(np.floor(bbox_center_y / grid_resolution_arr[1]))
			
			current_bbox_indexes = grid[row_index, column_index]
			current_bbox_indexes.append(box_index)
			grid[row_index, column_index] = current_bbox_indexes
			
			coord_pairs_arr[(row_index, column_index)] = current_bbox_indexes
		
		res = []
		# now go over all cells that do contain proposals
		for coord_pairs, bbox_indexes in coord_pairs_arr.items():
			
			x_left_arr = []
			x_right_arr = []
			y_top_arr = []
			y_bottom_arr = []
			
			for bbox_index in bbox_indexes:
				x_start, y_start, x_end, y_end = bbox_arr[bbox_index]
				
				x_left_arr.append(x_start)
				y_top_arr.append(y_start)
				x_right_arr.append(x_end)
				y_bottom_arr.append(y_end)
			
			bbox = [min(x_left_arr), min(y_top_arr), max(x_right_arr), max(y_bottom_arr)]
			res.append(bbox)
		
		return res
	
	# this function creates a list of bbox pairs that their bounding box (including padding) is smaller than the area of the sum of the two boxes
	def calc_candidate_pairs_for_merge(self, bbox_arr, valid_crop_arr, bbox_dist_matrix=None, index_to_recalculate=None, index_to_remove=None):
		# calculate IOU between all boxes
		
		# iou_arr = np.ones([len(bbox_arr), len(bbox_arr)]) * (-1)
		
		if bbox_dist_matrix is None:
			dist_arr = np.ones([len(bbox_arr), len(bbox_arr)]) * (100000)
		
		else:
			dist_arr = bbox_dist_matrix
		
		valid_arr = np.full([len(bbox_arr), len(bbox_arr)], False, dtype=bool)
		pair_data_arr = []
		
		for index1, bbox1 in enumerate(bbox_arr):
			for index2, bbox2 in enumerate(bbox_arr):
				
				if index1 != index2:
					
					if valid_crop_arr[index1] and valid_crop_arr[index2]:
						iou, area1, area2, intersection_area, unite_area = self.calc_iou(bbox1, bbox2)
						
						center_x1 = (bbox1[0] + bbox1[2]) / 2
						center_x2 = (bbox2[0] + bbox2[2]) / 2
						
						# iou_arr[index1, index2] = iou
						# iou_arr[index2, index1] = iou_arr[index1, index2]
						dist_arr[index1, index2] = abs(center_x1 - center_x2)
						dist_arr[index2, index1] = dist_arr[index1, index2]
						
						bounding_bbox, bounding_bbox_area = merge_utils.calc_bounding_box(bbox1, bbox2)
						
						if (area1 + area2) > bounding_bbox_area:
							valid_arr[index1, index2] = True
						
						# now check if s(A)+s(B) < s(bbox of A+B)
						
						# now sort pairs by highest iou, only for pairs that are worthwhile merging (valid pairs)
						
						if valid_arr[index1, index2]:
							pair_data_arr.append([dist_arr[index1, index2], index1, index2])
		
		pair_data_arr = np.array(pair_data_arr)
		
		return pair_data_arr
	
	# gets crops based on proposals and padding (determined per proposal height)
	# returns:
	#   bbox_arr: the suggested crop (based on proposal + padding)
	#   net_bbox_arr: the suggested crop (based on the proposal and image size)
	#   the original height of the proposal
	
	def get_crops(self, proposals_arr, inital_padding_arr, img_shape):
		# calculate initial bboxes
		bbox_arr = []
		net_bbox_arr = []
		crop_h_arr = []
		
		for proposal_index, bbox in enumerate(proposals_arr):
			
			center_x = (bbox[0] + bbox[2]) / 2
			center_y = (bbox[1] + bbox[3]) / 2
			
			proposal_half_height = bbox[3] - center_y
			proposal_half_width = bbox[2] - center_x
			
			proposal_height = proposal_half_height * 2
			crop_h_arr.append(proposal_height)
			
			# limit the proposals to the image size
			net_x_start = int(np.max([0, center_x - proposal_half_width]))
			net_x_end = int(np.min([img_shape[1] - 1, center_x + proposal_half_width]))
			net_y_start = int(np.max([0, center_y - proposal_half_height]))
			net_y_end = int(np.min([img_shape[0] - 1, center_y + proposal_half_height]))
			
			x_start = int(np.max([0, center_x - proposal_half_width - inital_padding_arr[1]]))
			y_start = int(np.max([0, center_y - proposal_half_height - inital_padding_arr[0]]))
			
			x_end = int(np.min([img_shape[1] - 1, center_x + proposal_half_width + inital_padding_arr[1]]))
			y_end = int(np.min([img_shape[0] - 1, center_y + proposal_half_height + inital_padding_arr[0]]))
			
			if (x_start < 0) or (y_start < 0) or (x_end >= img_shape[1]) or (y_end >= img_shape[0]):
				self.logger.error('Invalid crop coordinates!!! {}'.format(bbox))
				print('Invalid crop coordinates!!!{}'.format(bbox))
				return np.array([]), np.array([]), np.array([])
			
			net_bbox = np.array([net_x_start, net_y_start, net_x_end, net_y_end])
			bbox = np.array([x_start, y_start, x_end, y_end])
			
			bbox_arr.append(bbox)
			net_bbox_arr.append(net_bbox)
		
		return bbox_arr, net_bbox_arr, crop_h_arr
	
	# this method checks whether the given net bbox (without padding) can be included in the given allowed_size_arr bbox (without padding)
	# net_bbox format:[x_left, y_top, x_right, y_bottom]
	# net_allowed_size_arr: [h, w]
	def check_merge_validity(self, net_bbox, net_allowed_size_arr):
		
		net_bbox_width = net_bbox[2] - net_bbox[0]
		net_bbox_height = net_bbox[3] - net_bbox[1]
		
		ind = np.where((net_allowed_size_arr[:, 0] >= net_bbox_height) & ((net_allowed_size_arr[:, 1] >= net_bbox_width)))[0]
		if len(ind) > 0:
			return True
		
		return False
	
	# get the smallest crops that can entirely include the bbox, given the min_required_padding
	def get_best_fitted_crop(self, net_bbox, net_allowed_size_arr):
		net_bbox_width = net_bbox[2] - net_bbox[0]
		net_bbox_height = net_bbox[3] - net_bbox[1]
		
		diff_width_arr = net_allowed_size_arr[:, 1] - net_bbox_width
		diff_height_arr = net_allowed_size_arr[:, 0] - net_bbox_height
		
		ind = np.where((diff_width_arr < 0) | (diff_height_arr < 0))[0]
		
		if len(ind) == len(net_allowed_size_arr):
			print('Error:No valid crop was found------------------------------')
			return -1
		
		diff_width_arr[ind] = 10000
		diff_height_arr[ind] = 10000
		
		# get the tightest possible valid crop
		fitting_score = diff_width_arr * diff_height_arr
		
		ind_best = np.argmin(fitting_score)
		
		return ind_best
	
	# this method returns the coordinates of the crop given the net bbox, the crop size and the size of the given image
	
	def get_crop_with_padding(self, net_bbox, crop, img_shape):
		
		wanted_height = crop[0]
		wanted_width = crop[1]
		
		if (wanted_height > img_shape[0]) or (wanted_width > img_shape[1]):
			self.logger.error('crop [{},{}] bigger than given image shape [{}, {}]'.format(wanted_height, wanted_width, img_shape[0], img_shape[1]))
			return np.array([])
		
		x_left = net_bbox[0]
		y_top = net_bbox[1]
		x_right = net_bbox[2]
		y_bottom = net_bbox[3]
		
		net_bbox_height = y_bottom - y_top + 1
		net_bbox_width = x_right - x_left + 1
		
		x_padding = int((wanted_width - net_bbox_width) / 2)
		x_start = x_left - x_padding
		if x_start < 0:
			x_start = 0
		x_end = x_start + wanted_width - 1
		if x_end >= img_shape[1]:
			x_end = img_shape[1] - 1
			x_start = img_shape[1] - wanted_width
		
		y_padding = int((wanted_height - net_bbox_height) / 2)
		y_start = y_top - y_padding
		if y_start < 0:
			y_start = 0
		y_end = y_start + wanted_height - 1
		if y_end >= img_shape[0]:
			y_end = img_shape[0] - 1
			y_start = img_shape[0] - wanted_height
		
		if (x_start < 0) or (y_start < 0) or (x_end >= img_shape[1]) or (y_end >= img_shape[0]):
			print('Invalid crop coordinates!!!')
			return np.array([])
		
		crop_shape = [y_end - y_start + 1, x_end - x_start + 1]
		
		assert wanted_height == crop_shape[0]
		assert wanted_width == crop_shape[1]
		
		return np.array([x_start, y_start, x_end, y_end]).astype(int)
	
	# add temporary padding to net bbox for merge purposes
	def add_initial_padding(self, net_bbox, padding_arr, img_shape):
		
		x_left = net_bbox[0]
		y_top = net_bbox[1]
		x_right = net_bbox[2]
		y_bottom = net_bbox[3]
		
		x_start = int(np.max([0, x_left - padding_arr[1]]))
		y_start = int(np.max([0, y_top - padding_arr[0]]))
		
		x_end = int(np.min([img_shape[1] - 1, x_right + padding_arr[1]]))
		y_end = int(np.min([img_shape[0] - 1, y_bottom + padding_arr[0]]))
		
		if (x_start < 0) or (y_start < 0) or (x_end >= img_shape[1]) or (y_end >= img_shape[0]):
			print('Invalid crop coordinates!!!')
			return np.array([])
		
		return np.array([x_start, y_start, x_end, y_end]).astype(int)
	
	# this method merges proposals (called per height category)
	# crop_arr: list of crops (x1,y1, x2,y2)
	def get_merged_crops_per_height(self, crop_arr, net_crop_arr, allowed_size_arr, min_required_padding, inital_padding_arr, img_shape, img=[]):
		
		net_allowed_size_arr = []
		for crop in allowed_size_arr:
			new_height = crop[0] - 2 * min_required_padding[0]
			new_width = crop[1] - 2 * min_required_padding[1]
			
			if (new_height < 0) or (new_width < 0):
				self.logger.error('Given crop size [{}, {}] with padding [{}, {}] do not support padding'.format(crop[0], crop[1], new_height, new_width))
				return [], [], []
			
			net_allowed_size_arr.append([new_height, new_width])
		
		net_allowed_size_arr = np.array(net_allowed_size_arr)
		
		net_crop_height_arr = []
		for crop in net_crop_arr:
			net_crop_height_arr.append(crop[3] - crop[1])
		
		# ------------------------------------------------------------
		
		new_crop_arr = crop_arr.copy()
		new_net_crop_arr = net_crop_arr.copy()
		valid_crop_arr = []
		
		# index_arr stores the indexes of the original crops that participate in the merged crops
		index_arr = []
		for k in range(0, len(crop_arr)):
			index_arr.append([k])
			valid_crop_arr.append(True)
		
		# -------------------------------------------------------------
		
		pair_data_arr = self.calc_candidate_pairs_for_merge(crop_arr, valid_crop_arr)
		
		if len(pair_data_arr) > 0:
			
			while len(pair_data_arr) > 0:
				
				sort_index = pair_data_arr[:, 0].argsort()
				pair_data_arr = pair_data_arr[sort_index]
				
				index1 = int(pair_data_arr[0, 1])  # take the first pair
				index2 = int(pair_data_arr[0, 2])  # take the first pair
				
				bbox1_net = new_net_crop_arr[index1]
				bbox2_net = new_net_crop_arr[index2]
				
				# calculate net bounding box of two boxes
				bounding_net_bbox, bounding_bbox_area = merge_utils.calc_bounding_box(bbox1_net, bbox2_net)
				# add temporal padding to participate in merging
				bounding_bbox = self.add_initial_padding(bounding_net_bbox, inital_padding_arr, img_shape)
				
				# check if the merged bounding box is available for one of the propositions
				is_merge_valid = self.check_merge_validity(bounding_net_bbox, net_allowed_size_arr)
				
				# if we could not merge these crops
				if is_merge_valid is False:
					# exclude this crop from futher merging
					valid_crop_arr[index1] = False
					valid_crop_arr[index2] = False
				
				else:
					
					# if we are here we can add the bounding box
					# update crop_arr
					new_crop_arr[index1] = np.array(bounding_bbox)
					new_crop_arr2 = new_crop_arr[0:index2].copy()
					new_crop_arr2.extend(new_crop_arr[(index2 + 1):].copy())
					new_crop_arr = new_crop_arr2
					
					# update net_crop_arr
					new_net_crop_arr[index1] = np.array(bounding_net_bbox)
					new_net_crop_arr2 = new_net_crop_arr[0:index2].copy()
					new_net_crop_arr2.extend(new_net_crop_arr[(index2 + 1):].copy())
					new_net_crop_arr = new_net_crop_arr2
					
					# merge indexes
					index_arr[index1].extend(index_arr[index2])
					temp = index_arr[0:index2].copy()
					temp.extend(index_arr[(index2 + 1):].copy())
					index_arr = temp
					
					valid_crop_arr2 = valid_crop_arr[0:index2].copy()
					valid_crop_arr2.extend(valid_crop_arr[(index2 + 1):].copy())
					valid_crop_arr = valid_crop_arr2
					
					# debug per merge
					debug_result = False
					if debug_result:
						
						img_copy = np.zeros(img_shape)
						
						if len(img) > 0:
							img_copy = img.copy()
						
						for bbox_net in new_net_crop_arr:
							bbox_net = bbox_net.astype('int')
							
							cv2.rectangle(img_copy, (bbox_net[0], bbox_net[1]), (bbox_net[2], bbox_net[3]), color=(200, 200, 200), thickness=1)
						
						for index in index_arr[index1]:
							bbox_net = net_crop_arr[index].astype('int')
							bbox = crop_arr[index].astype('int')
							
							cv2.rectangle(img_copy, (bbox_net[0], bbox_net[1]), (bbox_net[2], bbox_net[3]), color=(255, 0, 255), thickness=1)
							cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 255), thickness=1)
						
						bounding_net_bbox = bounding_net_bbox.astype('int')
						bounding_bbox = bounding_bbox.astype('int')
						
						cv2.rectangle(img_copy, (bounding_net_bbox[0] - 5, bounding_net_bbox[1] - 5), (bounding_net_bbox[2] + 5, bounding_net_bbox[3] + 5), color=(0, 255, 0), thickness=1)
						cv2.rectangle(img_copy, (bounding_bbox[0] - 5, bounding_bbox[1] - 5), (bounding_bbox[2] + 5, bounding_bbox[3] + 5), color=(0, 0, 255), thickness=1)
						
						resized_img = cv2.resize(img_copy, (1200, 600), interpolation=cv2.INTER_LANCZOS4)
						cv2.imshow('', resized_img)
						print('show_debug_point')
				
				pair_data_arr = self.calc_candidate_pairs_for_merge(new_crop_arr, valid_crop_arr)
		
		# now add the needed padding to reach the real proposition dimensions
		
		res_crop_arr = []
		for net_crop_index, net_crop in enumerate(new_net_crop_arr):
			crop_index = self.get_best_fitted_crop(net_crop, net_allowed_size_arr)
			# TODO handle when crop_index = -1
			crop = self.get_crop_with_padding(net_crop, allowed_size_arr[crop_index], img_shape)
			if len(crop) > 0:
				res_crop_arr.append(crop)
		
		# # # # # # TODO remove this, for debug only
		# res_crop_arr = []
		# for net_crop_index, net_crop in enumerate(net_crop_arr):
		# 	crop_index = self.get_best_fitted_crop(net_crop, net_allowed_size_arr)
		# 	crop = self.get_crop_with_padding(net_crop, allowed_size_arr[crop_index], img_shape)
		# 	res_crop_arr.append(crop)
		
		return res_crop_arr, new_net_crop_arr, index_arr
	
	# main method
	# proposals (x1,y1,x2, y2) left-top, right-bottom corners
	# image object for debugging
	
	def pack_crops(self, proposals, img=[]):
		
		initial_crops_arr = np.empty([0, 5])  # the initial crops based on the given proposals
		initial_net_crops_arr = []  # the initial net crops (without padding) based on the given proposals
		initial_net_crops_per_height_arr = []  # the initial net crops (without padding) based on the given proposals, per height
		
		image_packed_crops_arr = np.empty([0, 5])  # the resulting crops per image after packing + weight factor
		new_net_crops_arr = []  # the net crops after packing
		
		# store also values per height group
		new_crops_per_height_arr = []  # packed crops per height
		new_crops_weight_per_height_arr = []  # crops weight (scale factor) per height
		
		proposal_height_arr = []
		
		# remove proposals with confidence smaller than defined
		if len(proposals) > 0:
			proposals = np.array(proposals)
			relevant_indexes = np.where(proposals[:, 4] >= self.proposals_min_conf)[0]
			proposals = proposals[relevant_indexes, :]
		else:
			proposals = []
		
		for proposal_index, proposal in enumerate(proposals):
			proposal_height = proposal[3] - proposal[1]
			
			proposal_height_arr.append(proposal_height)
		
		proposal_height_arr = np.array(proposal_height_arr)
		
		# for each height range process process proposals
		for h_index, h in enumerate(self.h_arr):
			
			h_min = h[0]
			h_max = h[1]
			scale_factor = self.scale_factor_arr[h_index]
			weight = 1 / scale_factor
			
			new_crops_per_height_arr.append([])
			new_crops_weight_per_height_arr.append([])
			initial_net_crops_per_height_arr.append([])
			
			# get the relevant proposals in this group class
			relevant_crop_indexes = np.where((proposal_height_arr >= h_min) & (proposal_height_arr < h_max))[0]
			proposals_per_height_arr_ = []
			
			for proposal_index in relevant_crop_indexes:
				proposal = proposals[proposal_index]
				
				x_left = proposal[0]
				x_right = proposal[2]
				y_top = proposal[1]
				y_bottom = proposal[3]
				
				proposals_per_height_arr_.append([x_left, y_top, x_right, y_bottom])
			
			# Do an initial merge of the proposals.
			try:
				proposals_per_height_arr = self.preprocess_proposals(proposals_per_height_arr_, self.merge_grid_resolution, self.img_shape)
				
				# Based on the proposals, get the crops.
				initial_crops_per_image_per_height, initial_net_crop_per_image_per_height, crop_h_arr_per_height = self.get_crops(proposals_per_height_arr, self.inital_padding_arr[h_index], self.img_shape)
				
				new_crops_per_height, new_net_crops_per_height, proposal_index_in_crop_arr = self.get_merged_crops_per_height(initial_crops_per_image_per_height, initial_net_crop_per_image_per_height,
				                                                                                                              self.allowed_size_arr[h_index], self.min_required_crop_padding_arr[h_index], self.inital_padding_arr[h_index],
				                                                                                                              self.img_shape, img)
				for crop_index, crop in enumerate(initial_net_crop_per_image_per_height):
					initial_net_crops_per_height_arr[h_index].append(crop.copy())
					initial_net_crops_arr.append(initial_net_crop_per_image_per_height[crop_index])
				
				for crop_index, crop in enumerate(initial_crops_per_image_per_height):
					init_crop = np.hstack([crop.copy().astype('int'), weight])
					if len(init_crop) > 0:
						initial_crops_arr = np.vstack([initial_crops_arr, init_crop])
				
				for crop_index, crop in enumerate(new_crops_per_height):
					if len(crop) == 0:
						print('error! crop is empty')
					
					new_net_crops_arr.append(crop.copy())
					new_crops_per_height_arr[h_index].append(crop.copy())
					new_crops_weight_per_height_arr[h_index].append(weight)
					
					updated_crop = np.hstack([crop.copy().astype('int'), weight])
					if len(updated_crop) > 0:
						image_packed_crops_arr = np.vstack([image_packed_crops_arr, updated_crop])
			
			except:
				print("Error")
		return image_packed_crops_arr, initial_crops_arr, initial_net_crops_arr, new_net_crops_arr, initial_net_crops_per_height_arr, new_crops_per_height_arr
