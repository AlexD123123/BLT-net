### BLT-net

Code for "You Better Look Twice: a new perspective for designing accurate detectors with reduced computations". BMVC, 2021.

### Brief description:
The code consists of the following main packages:
 - cascademv2: implementation of the CascadeMV2 first stage architecture.
  
           Configuration: The configuration of this architecture can be found in cascademv2/trainconfigs/
           
                 Due to many shared parameters, the training and testing configurations are defined in the same file, using different fields, if necessary. For more details see cascademv2/config_cascademv2.py
                 
           Input: The input for this module is located in data/input/cascademv2/data/Citypersons
           
                  To create the training/validation/test data please run the cascademv2/utils/generate_data_Citypersons.py
                  
           Training: to train the model using the configuration above run cascademv2/train.py
           
                Training output: trained models (saved per epoch)
                
           Evaluation: to evaluated the trained model using the configuration run cascade/eval_cascademv2.py
           
                    The testing process creates for each evaluated image a .txt file.
                    
                    All evaluated are then summarized into a jason file.
                    
                    To convert it to the pcmad input format please run cascademv2/utils/convert_cascademv2_json_to_pcmad_format.py  
                    

    Note: to run the cascademv2 cpu_nms, gpu_nms, bbox, cython_bbox should be complied.
    
           These can be done by running the makefiles from the original Faster R-CNN project.
 
 - PCMAD algorithm:
        This module merges and downscales the ROIs produced by the first stage.
        
        Run the algorithm from pcmad/run_pcmad.py
        
        The module also outputs relevant statistics.
        
        Configurations for PCMAD can be found in pcmad/pcmad_configs/                   
  
 - The output of the PCMAD algorithm can be supplied to any selected second-stage detector such as Pedestron, APD, ACSP.              
            





