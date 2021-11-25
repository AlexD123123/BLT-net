# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from blt_net.cascademv2.core.nms.gpu_nms import gpu_nms
from blt_net.cascademv2.core.nms import cpu_nms

def nms(dets, thresh, usegpu, gpu_id=0):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if usegpu:
        return gpu_nms(dets, thresh, device_id=gpu_id)
    else:
        return cpu_nms(dets, thresh)
