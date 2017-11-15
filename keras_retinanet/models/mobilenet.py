from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

import keras
from keras import backend as K
import keras_resnet.models
import keras_retinanet.models
from keras.applications.mobilenet import DepthwiseConv2D, _conv_block, _depthwise_conv_block, relu6, preprocess_input
import keras_retinanet.models
from keras.utils.data_utils import get_file

BASE_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/'

def MobileNet(inputs,
              alpha,
              depth_multiplier):

    img_input = inputs

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2) # 
    x0 = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x0, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x1 = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x1, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)6
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x2 = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x2, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)2
    x3 = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    model = keras.Model(inputs=inputs, outputs=[x0, x1, x2, x3])


    return model


def MobilenetRetinaNet(inputs, weights='imagenet', alpha=1.0, depth_multiplier=1, *args, **kwargs):

    if K.backend() != 'tensorflow':
        raise RuntimeError('Only TensorFlow backend is currently supported, '
                           'as other backends do not support '
                           'depthwise convolution.')


    if weights == 'imagenet':
        if depth_multiplier != 1:
            raise ValueError('If imagenet weights are being loaded, '
                             'depth multiplier must be 1')

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')

    image = inputs

    mobilenet = MobileNet(image, alpha=alpha, depth_multiplier=depth_multiplier)

    model = keras_retinanet.models.retinanet_bbox(inputs=inputs, backbone=mobilenet, *args, **kwargs)

    # load weights
    if weights == 'imagenet':
        rows = 224 # rows used in weight pretraining
        if K.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_last" format '
                             'are not available.')
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'

        model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
        weigh_path = BASE_WEIGHT_PATH + model_name
        weights_path = get_file(model_name, weigh_path, cache_subdir='models')
    else:
        weights_path = weights
    
    model.load_weights(weights_path, by_name=True)

    return model
