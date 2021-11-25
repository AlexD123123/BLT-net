from keras.layers import Input
from blt_net.cascademv2.core.utils import data_generators
import numpy as np


import os
from tensorflow.python.client import device_lib
import pickle


class Base_model():

    
    out_path = None
    
    def name(self):
        return 'BaseModel'
    
    def initialize(self, opt, logger):
        
        self.logger = logger
        self.num_gpus = len(opt.gpu_ids.split(','))
        self.init_lr = opt.init_lr
        
        self.num_epochs = opt.num_epochs
        self.add_epoch = opt.add_epoch
        
        self.epoch_length = -1 # to be defined later
        
        self.batch_size = opt.batch_size
        
        if opt.optimizer == 'Adam':
            from keras.optimizers import Adam
            self.optimizer = Adam(lr=opt.init_lr)
        elif opt.optimizer == 'AdamAcc':
            from blt_net.cascademv2.core.model import AdamAccumulate
            self.optimizer = AdamAccumulate(lr=opt.init_lr, accum_iters=opt.accum_iter)
        
        
        self.img_input = Input(shape=(opt.img_input[0], opt.img_input[1], 3))
        
        # To be created by create_backbone_model--------------------
        self.base_layers = []
        self.anchors = []
        self.num_anchors = []
        
        # To be created by set_data_loader------------------
        self.data_gen_train = None
        self.data_gen_val = None
        
        self.data_train_num_samples = None
        self.data_val_num_samples = None
        
        # To be created during training ----------------------
        self.train_total_loss = []
        self.val_total_loss = []
        
        # -------------------------------------------------
        self.out_path = opt.out_path
        os.makedirs(self.out_path, exist_ok=True)
        
        # -------------------------------------------------
        
        self.logger.info("----------------------------------------------------------------------------------------------------------------------------------")
        # print hardware details
        self.logger.info(device_lib.list_local_devices())
        
        # print configuration parameters
        #config_params = vars(opt)
        for item in opt:
            self.logger.info('{}:{}'.format(item, opt[item]))
    
    
    def create_backbone_model(self, opt):

        if opt.backbone == 'mobilenet2':
            from blt_net.cascademv2.core.model.backbones import mobilenet_v2 as nn
        else:
            raise NotImplementedError('Not support network: {}'.format(opt.network))

        # create the backbone (mobilenetv2)
        self.base_layers, feat_map_sizes = nn.nn_base(self.img_input, trainable=opt.train_backbone, logger=self.logger, opt=opt)
        # get default anchors and define data generator
        
        if opt.train_with_wma:
            self.base_layers_tea, _ = nn.nn_base(self.img_input, trainable=opt.train_backbone, logger=self.logger, opt=opt)
            
        self.anchors, self.num_anchors = data_generators.get_anchors(img_height=opt.img_input[0], img_width=opt.img_input[1],
                                                                     feat_map_sizes=feat_map_sizes.astype(np.int),
                                                                     anchor_box_scales=opt.anchor_box_scales,
                                                                     anchor_ratios=opt.anchor_ratios)
        
    
    def set_data_loader(self, opt, train_filename=None, val_filename=None, phase='train'):

        if phase == 'train':

            self.logger.info('Loading training data: {}'.format(train_filename))
            with open(train_filename, 'rb') as fid:
                train_data = pickle.load(fid)
            num_imgs_train = len(train_data)
            
            self.data_train_num_samples = len(train_data)
            
            self.epoch_length = int(self.data_train_num_samples / self.batch_size)
            
            self.logger.info('num of training samples: {}'.format(num_imgs_train))
            
            self.data_gen_train = data_generators.get_target(self.anchors, train_data, opt, batchsize=self.batch_size, net='2step',
                                                             igthre=opt.ig_overlap, posthre=opt.pos_overlap_step1,
                                                             negthre=opt.neg_overlap_step1)


            if val_filename:
                self.logger.info('Loading validation data: {}'.format(val_filename))
                with open(val_filename, 'rb') as fid:
                    val_data = pickle.load(fid)
                num_imgs_val = len(val_data)
                self.logger.info('num of validation samples: {}'.format(num_imgs_val))
                self.data_val_num_samples = len(val_data)

                self.data_gen_val = data_generators.get_target(self.anchors, val_data, opt, batchsize=self.batch_size, net='2step',
                                                               igthre=opt.ig_overlap, posthre=opt.pos_overlap_step1,
                                                               negthre=opt.neg_overlap_step1)


            


        

