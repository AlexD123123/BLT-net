from blt_net.cascademv2.core.model.base_model import Base_model
from keras.models import Model
from blt_net.cascademv2.core.model.parallel_model import ParallelModel
from keras.utils import generic_utils
from blt_net.cascademv2.core.model import losses as losses
from blt_net.cascademv2.utils import general_utils
import numpy as np
import time
import os
import cv2


class Model_2step(Base_model):
    
    def name(self):
        return 'Model_2step'
    
    def initialize(self, opt, logger):
        Base_model.initialize(self, opt, logger)
        
        self.train_cls_loss1 = []
        self.train_regr_loss1 = []
        self.train_cls_loss2 = []
        self.train_regr_loss2 = []
        
        self.val_cls_loss1 = []
        self.val_regr_loss1 = []
        self.val_cls_loss2 = []
        self.val_regr_loss2 = []
        
        self.logger.info('Initializing {}'.format(self.name()))
    
    def creat_model(self, opt,  phase='train'):

        Base_model.create_backbone_model(self, opt)
        
        if opt.head == 'cascademv2head':
            from blt_net.cascademv2.core.model.heads import cascademv2head as alf_head
        else:
            raise NotImplementedError('Not support network: {}'.format(opt.head))
        
        if phase == 'train':
            alf1, alf2 = alf_head.create_cascademv2(self.base_layers, self.num_anchors, trainable=True, steps=2, logger=self.logger)
        else:
            alf1, alf2 = alf_head.create_cascademv2(self.base_layers, self.num_anchors, trainable=False, steps=2, logger=self.logger)
        
        if opt.train_with_wma:
            if phase == 'train':
                alf1_tea, alf2_tea = alf_head.create_cascademv2(self.base_layers_tea, self.num_anchors, trainable=True, steps=2, logger=self.logger)
            else:
                alf1_tea, alf2_tea = alf_head.create_cascademv2(self.base_layers_tea, self.num_anchors, trainable=False, steps=2, logger=self.logger)
            
            self.model_tea = Model(self.img_input, alf1_tea + alf2_tea)
            
        if phase == 'train':
            self.model_1st = Model(self.img_input, alf1)
            self.model_2nd = Model(self.img_input, alf2)
            if self.num_gpus > 1:
                self.model_1st = ParallelModel(self.model_1st, int(self.num_gpus))
                self.model_2nd = ParallelModel(self.model_2nd, int(self.num_gpus))
            self.model_1st.compile(optimizer=self.optimizer, loss=[losses.cls_loss, losses.regr_loss], sample_weight_mode=None)
            self.model_2nd.compile(optimizer=self.optimizer, loss=[losses.cls_loss, losses.regr_loss], sample_weight_mode=None)
        self.model_all = Model(self.img_input, alf1 + alf2)
        
    # Counts the number of parameters in a model.
    def get_model_params(self):
        print(self.model_all.summary())
    
    # Removes the ALFNet head and saves only the backbone.
    def remove_head(self, model_filename, new_backbone_filename):
        
        self.model_all.load_weights(model_filename, by_name=True)
        self.logger.info('Loaded weights from {}'.format(model_filename))
        
        while len(self.model_all.layers) > 153:
            self.model_all.layers.pop(-1)
        
        self.model_all.save_weights(new_backbone_filename)
        
        print('')
    
    # Load the weights of a model.
    def load_model(self, weights_path):
        
        if self.logger is not None:
            self.logger.info('load weights from {}'.format(weights_path))
        
        # do not load multiple models by name
        self.model_all.load_weights(weights_path)

    # train model
    def train_model(self, opt, weight_path):
        
        from blt_net.cascademv2.core.utils import bbox_process
        
        if len(weight_path) > 0:
            
            load_by_name = False
            try:
                self.model_all.load_weights(weight_path,  by_name=load_by_name)
            except:
                load_by_name = True  # in case we load only the backbone
                self.model_all.load_weights(weight_path, by_name=load_by_name)
                
            if opt.train_with_wma:
                #if we are loading a pretrained model
                
                parts = weight_path.split('_e')
                if len(parts) == 2:
                    tea_weights_path = '{}_tea-e{}'.format(parts[0], parts[1])
                    if not os.path.isfile(tea_weights_path):
                        tea_weights_path = weight_path
                else:
                    tea_weights_path = weight_path

                self.model_tea.load_weights(tea_weights_path, by_name=load_by_name)
            self.logger.info('Loaded weights from {}'.format(weight_path))
        else:
            print('No pretrained model was defined. Training from scratch!')
        
        total_start_time = time.time()
        
        for epoch_num in range(opt.add_epoch + 1, self.num_epochs + 1):
            
            if opt.do_profiling:
                import cProfile
                
                pr = cProfile.Profile()
                pr.enable()
            
            epoch_losses = np.zeros((self.epoch_length, 6))
            progbar = generic_utils.Progbar(self.epoch_length)
            
            self.logger.info('Epoch {}/{}'.format(epoch_num, self.num_epochs))
            epoch_start_time = time.time()
            
            for iter_num in range(self.epoch_length):
                
                opt.current_epoch = epoch_num
        
                X, Y, img_data, orig_img_batch = next(self.data_gen_train)
                
                loss_s1 = self.model_1st.train_on_batch(X, Y)
                pred1 = self.model_1st.predict_on_batch(X)  # get regression values
                
                Y2 = bbox_process.get_target_1st(self.anchors, pred1[1], img_data, opt,
                                                 igthre=opt.ig_overlap, posthre=opt.pos_overlap_step2,
                                                 negthre=opt.neg_overlap_step2, input_image=orig_img_batch)
                
                # if an error has occured in get_target_1st method, ignore and continue
                if Y2 is None:
                    print('Error has occured in model_2step.. cannot predict Y2')
                    continue
                
                loss_s2 = self.model_2nd.train_on_batch(X, Y2)
                
                epoch_losses[iter_num, 0] = loss_s1[0]
                epoch_losses[iter_num, 1] = loss_s1[1]
                epoch_losses[iter_num, 2] = loss_s1[2]
                
                epoch_losses[iter_num, 3] = loss_s2[0]
                epoch_losses[iter_num, 4] = loss_s2[1]
                epoch_losses[iter_num, 5] = loss_s2[2]
                
                iter_num += 1
                
                total_loss1 = np.mean(epoch_losses[:iter_num, 0])
                cls_loss1 = np.mean(epoch_losses[:iter_num, 1])
                regr_loss1 = np.mean(epoch_losses[:iter_num, 2])
                
                total_loss2 = np.mean(epoch_losses[:iter_num, 3])
                cls_loss2 = np.mean(epoch_losses[:iter_num, 4])
                regr_loss2 = np.mean(epoch_losses[:iter_num, 5])
                
                total_loss = total_loss1 + total_loss2
                
                if opt.train_with_wma:
                    # Apply weight moving average
                    for layer_index in range(len(self.model_tea.layers)):
                        weights_tea = self.model_tea.layers[layer_index].get_weights()
                        weights_stu = self.model_all.layers[layer_index].get_weights()
                        if len(weights_stu) > 0:
                            weights_tea = [opt.wma_alpha * w_tea + (1 - opt.wma_alpha) * w_stu for (w_tea, w_stu) in
                                       zip(weights_tea, weights_stu)]
                            
                            self.model_tea.layers[layer_index].set_weights(weights_tea)
                            
                if iter_num % 20 == 0:
                    progbar.update(iter_num,
                                   [('train_total', total_loss),
                                    ('cls1', cls_loss1),
                                    ('regr1', regr_loss1),
                                    ('cls2', cls_loss2),
                                    ('regr2', regr_loss2)])
            
            if opt.do_profiling:
                pr.disable()
                # after your program ends
                pr.print_stats(sort="cumtime")
                print('end profiling for this epoch')
            
            epoch_runtime = time.time() - epoch_start_time
            self.logger.info("Training time: {:1.2f} [s]".format(epoch_runtime))
            
            self.train_total_loss.append(total_loss)
            self.train_cls_loss1.append(cls_loss1)
            self.train_regr_loss1.append(regr_loss1)
            self.train_cls_loss2.append(cls_loss2)
            self.train_regr_loss2.append(regr_loss2)
            
            self.logger.info("Training total/cls/reg loss: {:1.3f}/{:1.3f}/{:1.3f}/{:1.3f}/{:1.3f}".format(total_loss, cls_loss1, regr_loss1, cls_loss2, regr_loss2))
            
            if (self.data_gen_val is not None) and (epoch_num % opt.val_epoch_step == 0):
                
                epoch_val_total_loss, epoch_val_cls_loss1, epoch_val_regr_loss1, epoch_val_cls_loss2, epoch_val_regr_loss2 = self.calc_performance_error(opt, self.data_gen_val, self.data_val_num_samples)
                self.val_total_loss.append(epoch_val_total_loss)
                self.val_cls_loss1.append(epoch_val_cls_loss1)
                self.val_regr_loss1.append(epoch_val_regr_loss1)
                self.val_cls_loss2.append(epoch_val_cls_loss2)
                self.val_regr_loss2.append(epoch_val_regr_loss2)
                self.logger.info("Validation total/cls/reg loss: {:1.3f}/{:1.3f}/{:1.3f}/{:1.3f}/{:1.3f}".format(epoch_val_total_loss, epoch_val_cls_loss1, epoch_val_regr_loss1,
                                                                                                                 epoch_val_cls_loss2, epoch_val_regr_loss2))
                
                records = np.concatenate((np.asarray(self.train_total_loss).reshape((-1, 1)),
                                          np.asarray(self.train_cls_loss1).reshape((-1, 1)),
                                          np.asarray(self.train_regr_loss1).reshape((-1, 1)),
                                          np.asarray(self.train_cls_loss2).reshape((-1, 1)),
                                          np.asarray(self.train_regr_loss2).reshape((-1, 1)),
                                          np.asarray(self.val_total_loss).reshape((-1, 1)),
                                          np.asarray(self.val_cls_loss1).reshape((-1, 1)),
                                          np.asarray(self.val_regr_loss1).reshape((-1, 1)),
                                          np.asarray(self.val_cls_loss2).reshape((-1, 1)),
                                          np.asarray(self.val_regr_loss2).reshape((-1, 1))),
                                         axis=-1)
            
            else:
                
                # Stub values if for this epoch we don't have a validation error result
                self.val_total_loss.append(-1)
                self.val_cls_loss1.append(-1)
                self.val_regr_loss1.append(-1)
                self.val_cls_loss2.append(-1)
                self.val_regr_loss2.append(-1)
                
                records = np.concatenate((np.asarray(self.train_total_loss).reshape((-1, 1)),
                                          np.asarray(self.train_cls_loss1).reshape((-1, 1)),
                                          np.asarray(self.train_regr_loss1).reshape((-1, 1)),
                                          np.asarray(self.train_cls_loss2).reshape((-1, 1)),
                                          np.asarray(self.train_regr_loss2).reshape((-1, 1)),
                                          np.asarray(self.val_total_loss).reshape((-1, 1)),
                                          np.asarray(self.val_cls_loss1).reshape((-1, 1)),
                                          np.asarray(self.val_regr_loss1).reshape((-1, 1)),
                                          np.asarray(self.val_cls_loss2).reshape((-1, 1)),
                                          np.asarray(self.val_regr_loss2).reshape((-1, 1))),
                                         axis=-1)
            
            model_name = '{}_e{}_tl{:1.3f}_vl{:1.3f}.hdf5'.format(opt.backbone, epoch_num, self.train_total_loss[-1], self.val_total_loss[-1])
            self.model_all.save_weights(os.path.join(self.out_path, model_name))
            
            
            if opt.train_with_wma:
                teacher_model_name = '{}_tea-e{}_tl{:1.3f}_vl{:1.3f}.hdf5'.format(opt.backbone, epoch_num, self.train_total_loss[-1], self.val_total_loss[-1])
                self.model_tea.save_weights(os.path.join(self.out_path, teacher_model_name))
                
            np.savetxt(os.path.join(self.out_path, 'records.txt'), np.array(records), fmt='%.4f')
            
            self.logger.info("\nTotal model training time: {:1.2f} [s]\n".format(time.time() - total_start_time))
    
        self.logger.info('Training complete, exiting.')


    # Calc performance error
    def calc_performance_error(self, opt, data_gen, num_samples):
        from blt_net.cascademv2.core.utils import bbox_process
        
        num_epochs = int(num_samples / self.batch_size)
        
        # calc validation loss per epoch
        epoch_losses = np.zeros((num_epochs, 6))
        iter_num = 0
        
        while True:
            try:
                
                X, Y, img_data, orig_img_batch = next(data_gen)
                val_loss1 = self.model_1st.test_on_batch(X, Y)
                pred1 = self.model_1st.predict_on_batch(X)
                
                Y2 = bbox_process.get_target_1st(self.anchors, pred1[1], img_data, opt,
                                                 igthre=opt.ig_overlap, posthre=opt.pos_overlap_step2,
                                                 negthre=opt.neg_overlap_step2)
                
                val_loss2 = self.model_2nd.test_on_batch(X, Y2)
                
                epoch_losses[iter_num, 0] = val_loss1[0]
                epoch_losses[iter_num, 1] = val_loss1[1]
                epoch_losses[iter_num, 2] = val_loss1[2]
                
                epoch_losses[iter_num, 3] = val_loss2[0]
                epoch_losses[iter_num, 4] = val_loss2[1]
                epoch_losses[iter_num, 5] = val_loss2[2]
                
                iter_num += 1
                
                if iter_num == num_epochs:
                    break
            
            except Exception as e:
                self.logger.info('Exception in model_2step.py (calc_performance_error()): {}'.format(e))
                continue
        
        # update validation scores
        total_loss1 = np.mean(epoch_losses[:iter_num, 0])
        cls_loss1 = np.mean(epoch_losses[:iter_num, 1])
        regr_loss1 = np.mean(epoch_losses[:iter_num, 2])
        
        total_loss2 = np.mean(epoch_losses[:iter_num, 3])
        cls_loss2 = np.mean(epoch_losses[:iter_num, 4])
        regr_loss2 = np.mean(epoch_losses[:iter_num, 5])
        
        total_loss = total_loss1 + total_loss2
        
        return total_loss, cls_loss1, regr_loss1, cls_loss2, regr_loss2

    def get_dt_min_max_anchor_width(self, img, opt):
        if len(opt.dt_min_anchor_width) > 0:
            
            min_anchor_width_data = np.array(opt.dt_min_anchor_width)
            min_crop_height_arr = min_anchor_width_data[:, 0]
            max_crop_height_arr = min_anchor_width_data[:, 1]
            min_anchor_width_arr = min_anchor_width_data[:, 2]
            
            ind = np.where(np.logical_and(img.shape[0] >= min_crop_height_arr, img.shape[0] <= max_crop_height_arr))[0]
            
            if len(ind) > 0:
                dt_min_anchor_width = min_anchor_width_arr[ind[0]]
            else:
                dt_min_anchor_width = 0
        
        else:
            dt_min_anchor_width = 0
        
        if len(opt.dt_max_anchor_width) > 0:
            
            max_anchor_width_data = np.array(opt.dt_max_anchor_width)
            min_crop_height_arr = max_anchor_width_data[:, 0]
            max_crop_height_arr = max_anchor_width_data[:, 1]
            max_anchor_width_arr = max_anchor_width_data[:, 2]
            
            ind = np.where(np.logical_and(img.shape[0] >= min_crop_height_arr, img.shape[0] <= max_crop_height_arr))[0]
            
            if len(ind) > 0:
                dt_max_anchor_width = max_anchor_width_arr[ind[0]]
            else:
                dt_max_anchor_width = 10000
        else:
            dt_max_anchor_width = 10000
            
        return dt_min_anchor_width, dt_max_anchor_width
    
    # Infer model on images.
    # returns:
    #   res_per_image: nparray of detections x_left,y_top,w,h,detection confidence, anchors_l2, anchors_l1
    #   infer_time
    
    def test_model(self, opt, img, gt_arr=[], res_txt_filename='', res_img_filename=''):
        
        from blt_net.cascademv2.core.utils import bbox_process
        
        wanted_input_size = opt.img_input
        img_resized = cv2.resize(img, (wanted_input_size[1], wanted_input_size[0]), interpolation=cv2.INTER_LANCZOS4)
        
        # img_resized = img
        start_time = time.time()
        
        x_in = bbox_process.format_img(img_resized, opt)
        
        Y = self.model_all.predict(x_in)
        
        # Get estimations relative to anchors of step1 predictions (Y[0] are the scores, Y[1] are the regression values).
        proposals = bbox_process.pred_pp_1st(self.anchors, Y[0], Y[1], opt)
        
        dt_min_anchor_width, dt_max_anchor_width = self.get_dt_min_max_anchor_width(img, opt)
        
        
        # Get estimations relative to predictions from step1 (Y[2] are the scores, Y[3] are the regression values).
        bbxs, scores, proposals_pre_nms, scores_pre_nms, anchors, anchors_l1 = bbox_process.pred_det(proposals, Y[2], Y[3], opt, step=2,
                                                                                                     anchors_l1=self.anchors,
                                                                                                     dt_min_anchor_width=dt_min_anchor_width, dt_max_anchor_width=dt_max_anchor_width)
        
        end_time = time.time()
        
        infer_time = end_time - start_time
        
        # Save upper left corner and width and height.
        bbxs[:, [2, 3]] -= bbxs[:, [0, 1]]
        
        anchors[:, [2, 3]] -= anchors[:, [0, 1]]
        anchors_l1[:, [2, 3]] -= anchors_l1[:, [0, 1]]
        
        # Now filter detections by height.
        height_arr = bbxs[:, 3]
        keep = np.where(height_arr>=opt.pd_min_height)[0]
        bbxs = bbxs[keep, :]
        scores = scores[keep]
        anchors = anchors[keep, :]
        anchors_l1 = anchors_l1[keep, :]
        
        res_per_image = np.concatenate((bbxs, scores, anchors, anchors_l1), axis=-1).tolist()
        
        if opt.create_image_with_gt:
            if len(gt_arr) > 0:
                gt_arr = np.array(gt_arr)
                gt_arr[:, 2] = (gt_arr[:, 2] - gt_arr[:, 0])
                gt_arr[:, 3] = (gt_arr[:, 3] - gt_arr[:, 1])
        else:
            gt_arr = []
        
        # If create image option
        if opt.create_image or opt.show_image:
            general_utils.createImage(img, bbxs, scores, res_img_filename, gt_arr, show_image=opt.show_image)

        if len(res_txt_filename)>0:
            np.savetxt(res_txt_filename, np.array(res_per_image), fmt='%.4f')
        
        return res_per_image, infer_time
        
    # Test model on a single crop (image could be resized)
    def test_model_on_crop(self, img_crop, opt, dt_min_anchor_width=0, dt_max_anchor_width=10000, remove_overlapping_detections=False):
        
        from blt_net.cascademv2.core.nms import cascademv2_nms
        from blt_net.cascademv2.core.utils import bbox_process
        
        x_in = np.expand_dims(img_crop, axis=0)
        
        Y = self.model_all.predict(x_in)
        
        proposals = bbox_process.pred_pp_1st(self.anchors, Y[0], Y[1], opt)
        
        bbxs, scores, bbxs_pre_nms, scores_pre_nms, anchors, anchors_l1 = bbox_process.pred_det(proposals, Y[2], Y[3], opt, step=2,
                                                                                                anchors_l1=self.anchors,
                                                                                                dt_min_anchor_width=dt_min_anchor_width,
                                                                                                dt_max_anchor_width=dt_max_anchor_width)
        
        # In case we want to remove overlapping small detections to big detection
        if remove_overlapping_detections:
        
            anchor_width_l1 = anchors_l1[:, 2] - anchors_l1[:, 0]
            
            anchor_width_l1 = np.reshape(anchor_width_l1, [len(anchor_width_l1), 1])
 
            candidate_indexes = np.where(np.logical_and(anchor_width_l1 < opt.dt_max_anchor_width_for_removal, scores<=opt.dt_max_confidence_for_removal))[0]
            
            if len(candidate_indexes)>0 and len(bbxs)>0:
                bad_indexes = cascademv2_nms.remove_small_dets(bbxs, scores, candidate_indexes_arr=candidate_indexes,
                                                               dt_min_height_background=opt.dt_min_height_background,
                                                               dt_min_confidence_for_background=opt.dt_min_confidence_for_background,
                                                               dt_min_area_overlap_ratio=opt.dt_min_area_overlap_ratio)
            
                if len(bad_indexes):
                    all_indexes = np.array(range(len(bbxs)))
                    keep = np.delete(all_indexes, bad_indexes)
                    
                    bbxs = bbxs[keep, :]
        
                    anchors = anchors[keep, :]
                    anchors_l1 = anchors_l1[keep, :]
        
        return bbxs, scores, bbxs_pre_nms, scores_pre_nms, anchors, anchors_l1
    

    # save keras model as tf model
    # weight_filename: the .h5 filename
    # out_path: output directory
    # model_name: model_name (without extension)

    def config_keras_backend(self):
        import tensorflow as tf
        """Config tensorflow backend to use less GPU memory."""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        session = tf.Session(config=config)
        tf.keras.backend.set_session(session)
        
        
    def save_chkpt(self, weight_filename, check_point_filename):
        
        import tensorflow as tf
        from keras import backend as K
        self.model_all.load_weights(weight_filename) #by_name=False
        print('load weights from {}'.format(weight_filename))
        
        self.model_all.layers[0].name = 'input'
        self.model_all.summary()
        
        sess = K.get_session()

        saver = tf.compat.v1.train.Saver()

        saver.save(sess, check_point_filename)
        
    # Print keras model summary.
    def summary(self):
        self.model_all.summary()
    