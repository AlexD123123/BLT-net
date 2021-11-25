from yacs.config import CfgNode as CN


def get_cfg_defaults():
	"""Get a yacs CfgNode object with default values for my_project."""
	# Return a clone so that the defaults will not be altered
	# This is for the "local variable" use pattern
	return _C.clone()


_C = CN()

_C.SYSTEM = CN()

_C.config_name = 'roi_config_default'

_C.img_shape = [1024, 2048]

_C.merge_grid_resolution = [20, 20]

_C.allowed_size_arr = [[[256, 128], [256, 256], [256, 384], [256, 512]],
                       [[384, 192], [384, 384], [384, 576], [384, 768]],
                       [[512, 256], [512, 512], [512, 768], [512, 1024]],
                       [[640, 320], [640, 640], [640, 960], [640, 1280]],
                       [[768, 384], [768, 768], [768, 1152], [768, 1536]]]

_C.scale_factor_arr = [1, 1.5, 2, 2.5, 3]

#01.01.2020
# _C.inital_padding_arr = [[20, 40], [20, 40], [40, 40], [50, 50], [60, 60]]
# _C.min_required_crop_padding_arr = [[30, 30], [30, 30], [40, 40], [50, 50], [60, 60]]

#28.03.2020
_C.inital_padding_arr = [[20, 40], [20, 40], [60, 60], [60, 60], [80, 80]]
_C.min_required_crop_padding_arr = [[30, 30], [30, 30], [60, 60], [60, 60], [80, 80]]


_C.proposals_min_conf = 0.01

