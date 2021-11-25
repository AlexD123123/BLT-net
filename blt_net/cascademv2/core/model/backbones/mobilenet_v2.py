"""MobileNet v2 models for Keras.
# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)

adopted from:
https://github.com/xiaochus/MobileNetV2/blob/master/mobilenet_v2.py
"""

#from keras.applications.mobilenetv2 import MobileNetV2


from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
from keras import backend as K
import numpy as np

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# f(x) = min(max(x, 0),6)
def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)


def _conv_block(inputs, filters, kernel, strides, trainable=False):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides,  trainable=trainable)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False,  trainable=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        alpha: Integer, width multiplier. If apha <1 it proportionally decreases the number of filters in each layer. If alpha>1 it increases the number of filters in each layer
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Depth
    tchannel = K.int_shape(inputs)[channel_axis] * t
    # Width
    cchannel = int(filters * alpha)

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1), trainable=trainable)

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same',  trainable=trainable)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same', trainable=trainable)(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x


def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n, trainable=False):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        alpha: Integer, width multiplier.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, alpha, strides, trainable=trainable)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, alpha, 1, True, trainable=trainable)

    return x


def nn_base(img_input, alpha=1.0, trainable=True, logger=None, opt=None):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
        alpha: Integer, width multiplier, better in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4].
    # Returns
        MobileNetv2 model.
    """
    #inputs = Input(shape=input_shape)

    #img_input = None,

    first_filters = _make_divisible(32 * alpha, 8)
    
    x = _conv_block(img_input, first_filters, (3, 3), strides=(2, 2),  trainable=False) #block_id = 0
    
    
    train_layer = True if trainable and (opt.train_backbone_level<=1) else False
    x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1, trainable=train_layer) #block_id = 1
    
    train_layer = True if trainable and (opt.train_backbone_level<=2) else False
    stage2 = _inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2, trainable=train_layer) #block_id = 2
    
    train_layer = True if trainable and (opt.train_backbone_level <= 3) else False
    stage3 = _inverted_residual_block(stage2, 32, (3, 3), t=6, alpha=alpha, strides=2, n=3, trainable=train_layer) #block_id = 3 (C3)
    
    train_layer = True if trainable and (opt.train_backbone_level <= 4) else False
    x = _inverted_residual_block(stage3, 64, (3, 3), t=6, alpha=alpha, strides=2, n=4, trainable=train_layer) #block_id = 4
    
    train_layer = True if trainable and (opt.train_backbone_level <= 5) else False
    stage4 = _inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=3, trainable=train_layer) #block_id = 5 (C4)
    
    train_layer = True if trainable and (opt.train_backbone_level <= 6) else False
    x = _inverted_residual_block(stage4, 160, (3, 3), t=6, alpha=alpha, strides=2, n=3, trainable=train_layer)  #block_id = 6
    
    train_layer = True if trainable and (opt.train_backbone_level <= 7) else False
    x = _inverted_residual_block(x, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1, trainable=train_layer)  #block_id = 7

    if alpha > 1.0:
        last_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_filters = 1280
    
    train_layer = True if trainable and (opt.train_backbone_level <= 8) else False
    stage5 = _conv_block(x, last_filters, (1, 1), strides=(1, 1), trainable=train_layer) #block_id = 8 (C5)
    
    if logger is not None:
        logger.info('Input: {}'.format(img_input._keras_shape[1:]))
        logger.info('C2: {}'.format(stage2._keras_shape[1:]))
        logger.info('C3: {}'.format(stage3._keras_shape[1:]))
        logger.info('C4: {}'.format(stage4._keras_shape[1:]))
        logger.info('C5: {}'.format(stage5._keras_shape[1:]))
    
    
    # Does not use these mv2 layers
    #
    #x = GlobalAveragePooling2D()(stage5, trainable=trainable)
    #x = Reshape((1, 1, last_filters))(x)
    #x = Dropout(0.3, name='Dropout')(x)
    #x = Conv2D(k, (1, 1), padding='same')(x)

    #x = Activation('softmax', name='softmax')(x)
    #output = Reshape((k,))(x)

    #model = Model(inputs, output)
    # plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)

    
    predictor_sizes = np.array([stage3._keras_shape[1:3],
                                stage4._keras_shape[1:3],
                                stage5._keras_shape[1:3],
                                np.ceil(np.array(stage5._keras_shape[1:3]) / 2)])
    return [stage3, stage4, stage5, stage2], predictor_sizes