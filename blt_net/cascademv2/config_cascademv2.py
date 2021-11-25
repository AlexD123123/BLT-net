from yacs.config import CfgNode as CN


def get_cfg_defaults():
	"""Get a yacs CfgNode object with default values for my_project."""
	# Return a clone so that the defaults will not be altered
	# This is for the "local variable" use pattern
	return _C.clone()


_C = CN()

_C.SYSTEM = CN()

# general parameters --------------------------------------------
_C.config_name = 'config_cascademv2'
_C.gpu_ids = 'cuda:0'  # '2'

# setting for network architechture

_C.backbone = 'mobilenet2'  # 'mobilenet1','mobilenet2' ,  'resnet50', mobilenet2gn
#relevant only for mobilenetg2_groupnorm
_C.num_groups = 1 #1: layer normalization

_C.head = 'cascademv2head'
_C.steps = 2

# Defines the input of the network.
# This parameter is set at run time. For training it is set as the value of _C.random_crop and in eval as _C.eval_im_input
_C.img_input = (-1, -1)

# other infrastructure flags
_C.framework='keras'

# image channel-wise mean to subtract, the order is BGR (info. used for training and infer mode, relevant for Cityperson dataset)
_C.img_channel_mean = [103.939, 116.779, 123.68]

#anchors information-----------------------------------------------
# setting for scales
# anchor boxes size
_C.anchor_box_scales = [[16, 24], [32, 48], [64, 80], [128, 160]]

# aspect ratio for each scale
_C.anchor_ratios = [[0.41], [0.41], [0.41], [0.41]]

# scaling the stdev
_C.classifier_regr_std = [0.1, 0.1, 0.2, 0.2]


# train parameters-----------------------------------------------------------------------

# profiling parameters
_C.do_profiling = False

#use mean average weights teacher
_C.train_with_wma = False
_C.wma_alpha = 0.999 #weight moving average alpha

#whether to train also the backbone or to freeze its weights
_C.train_backbone = True
_C.train_backbone_level = 1 #from level 1 and upwards

#pretrained mobilnet v2

_C.initial_model = ''  # load last model
_C.add_epoch = -1 #set as -1 to continue training from last existing model. Set add_epoch>0 to start from a specific epoch. If no model is found train from initial model

_C.out_path = '../data/output/models/Citypersons_'

_C.train_filename = '../data/input/cascademv2/data/Citypersons/train'
_C.val_filename = '../data/input/cascademv2/data/Citypersons/val'

_C.batch_size = 4  # 1 #4 #batch size
_C.num_epochs = 350

_C.init_lr = 1e-4

_C.optimizer = 'Adam'

_C.accum_iter = 5 # relevant only for AdamAcc

#input size for training
_C.random_crop = (640, 1280)

#-----------------------------------------------------------------

# setting augmentation parameters (in training)
_C.augmentBrightness = True
_C.augmentCrop = True

# setting for data augmentation
#augmentation of crop: <1 means we take a smaller image and upscale to crop size; > 1 means we take a bigger image and downscale to crop size
_C.scale = (0.3, 1.0)
_C.use_horizontal_flips = True

_C.brightness = (0.5, 2, 0.5)

#------------------------------------------------


# overlaps IOU criterion for ignore areas
_C.ig_overlap = 0.5

# overlaps for different ALF steps
_C.neg_overlap_step1 = 0.3
_C.pos_overlap_step1 = 0.5
_C.neg_overlap_step2 = 0.4
_C.pos_overlap_step2 = 0.65
_C.neg_overlap_step3 = 0.5
_C.pos_overlap_step3 = 0.75


#settings for inference on full image --------------------------------------------------------------------------
_C.eval_im_input = (1024, 2048) #image input size, determines the network input size

_C.eval_dataset_type = 'Citypersons'
_C.eval_filename = '../data/input/cascademv2/data/Citypersons/val'  # infer the model on these images

_C.eval_output_root = ""
_C.model_name = '' # model name to use for inference

_C.eval_min_epoch = 200 #min epoch to start evaluating from
_C.eval_resolution_step = 1 #evaluation step
_C.eval_on_val = True #if true then output is created in the 'val' folder otherwise in the 'train' folder

_C.scorethre = 0.1  # Consider windows only over this threshold to be considered for NMS
_C.overlap_thresh = 0.5 # Minimum overlap between detections to be merged by NMS
_C.pre_nms_topN = 6000
_C.post_nms_topN = 100 # Take only this top predictions
_C.roi_stride = 16 # This is the mininum acceptable height/width of the proposals  (16/0.41=39 px), everything else is filtered. Activated on the output of the network

_C.dt_min_anchor_width = [[0, 5000, 0]] #format: min_image height, max_image_height, min_anchor_width i.e. [[0, 256, 0], [257, 10000, 25]]
_C.dt_max_anchor_width = [[0, 5000, 1000]] #format: min_image height, max_image_height, max_anchor_width i.e. [[0, 256, 0], [257, 10000, 25]]

_C.pd_min_height = 16

# image creation parameters----------------------------------

_C.create_image_with_gt = False # Create image with GTs

_C.show_pre_nms = False

_C.create_image = False # Create image and output it to the effective output_path
_C.show_image = False # show images (does not save them)

