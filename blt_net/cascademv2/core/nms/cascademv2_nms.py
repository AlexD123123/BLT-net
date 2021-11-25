import numpy as np

#candidate_indexes_arr: indexes of candidate detections to remove
#min_area_overlap_ratio: maximum overlap ratio between candidate bbox and others bbox that beyond that cadidate box is removed
#   1: the candidate bbox should overlap 100% of its area with a bg bbox to be removed
def remove_small_dets(dets, scores, candidate_indexes_arr, dt_min_height_background=0,
                      dt_min_confidence_for_background=0.5, dt_min_area_overlap_ratio=1):
	
	removed_indexes_arr = []
	
	for ind_candidate in candidate_indexes_arr:
		#calculate overlap with another DT
	
		candidate_det = dets[ind_candidate, :]
		
		candidate_dt_height = candidate_det[3] - candidate_det[1]
		
		for det_index, other_det in enumerate(dets):
			
			other_det_height = other_det[3] - other_det[1]
			
			# if the background dt is relevant (i.e. smaller than the bg bbox and the bg bbox is higher than a certain threshold)
			if (candidate_dt_height < other_det_height) and (other_det_height >= dt_min_height_background) and (scores[det_index]>=dt_min_confidence_for_background):
			#if (candidate_dt_height < other_det_height) and (other_det_height >= dt_min_height_background):
				
				x_start1, y_start1, x_end1, y_end1 = candidate_det
				x_start2, y_start2, x_end2, y_end2 = other_det
		
				x_left = max(x_start1, x_start2)
				y_top = max(y_start1, y_start2)
				x_right = min(x_end1, x_end2)
				y_bottom = min(y_end1, y_end2)
				
				# if there is no overlap
				if (x_right < x_left) or (y_bottom < y_top):
					intersection_area = 0
				
				else:
					# The intersection of two axis-aligned bounding boxes is always an
					# axis-aligned bounding box
					intersection_area = (x_right - x_left) * (y_bottom - y_top)
				
				bb1_area = (x_end1 - x_start1) * (y_end1 - y_start1)
				bb2_area = (x_end2 - x_start2) * (y_end2 - y_start2)
					
				bb1_cover_area_ratio = intersection_area/bb1_area
				
				if bb1_cover_area_ratio > dt_min_area_overlap_ratio:
					removed_indexes_arr.append(ind_candidate)
	
	removed_indexes_arr = np.array(removed_indexes_arr)
	
	return np.unique(removed_indexes_arr)