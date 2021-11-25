import json
import os
import yaml
import pickle
import numpy as np

def convert_results(results_path, dataset_type):
	if dataset_type == 'Citypersons':
		convert_Citypersons_results(results_path)
	else:
		print('Cannot convert  dataset type: {}. Returning...'.format(dataset_type))

def convert_Citypersons_results(results_path, is_validation=True):
	
	
	numImages = 500
	file = open('../../data/input/cascademv2/data/Citypersons/val_according_to_GT.txt', 'r')
	
	imNames = file.readlines()
	newImNames = []
	for line in imNames:
		lineSplit = line.split('/')
		lineSplit = lineSplit[len(lineSplit) - 1]
		lineSplit = lineSplit.split('_leftImg8bit')
		lineSplit = lineSplit[0]
		newImNames.append(lineSplit)
	
	resultsFile = open(os.path.join(results_path, 'all_results.txt'), 'w')
	
	for k in range(0, numImages):
		
		try:
			file2 = open(os.path.join(results_path, newImNames[k] + '_leftImg8bit.txt'), 'r')
			data = file2.readlines()
			
			for i in range(0, len(data)):
				parsedRow = data[i].split("\n")
				parsedRow = parsedRow[0]
				parsedRow = parsedRow.split(" ")
				resultsFile.write(str(k + 1) + " " + parsedRow[0] + " " + parsedRow[1] + " " + parsedRow[2] + " " + parsedRow[3] + " " + parsedRow[4] + "\n")
		except:
			print('convertAlfNetDetectionsResults(): error in processing {}'.format(newImNames[k]))
	
	resultsFile.close()
	
	filename = os.path.join(results_path, 'all_results.txt')
	
	data = []
	with open(filename) as fh:
		for line in fh:
			im_id, bbox1, bbox2, bbox3, bbox4, confidence = line.strip().split(' ')
			imIdVec = int(im_id)
			# bboxVec = [float(bbox1), float(bbox2), float(bbox3), float(bbox4)]
			bboxVec = [float(bbox1), float(bbox2), float(bbox3), float(bbox4)]
			confVec = float(confidence)
			curData = {}
			curData['image_id'] = imIdVec
			curData['bbox'] = bboxVec
			curData['confidence'] = confVec
			curData['score'] = confVec
			curData['category_id'] = 1
			data.append(curData)
	
	filenameJson = os.path.join(results_path, 'all_results.json')
	with open(filenameJson, 'w') as f:
		json.dump(data, f)

	

