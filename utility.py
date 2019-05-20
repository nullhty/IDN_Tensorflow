"""
Created on 2019-04-29
@author: hty
"""
import tensorflow as tf
 
def weight_variable(shape, name):
    weight = tf.get_variable(name, shape, initializer=tf.keras.initializers.he_normal(), dtype=tf.float32)
    return weight
    
    
def bias_variable(shape, name):
    bias = tf.get_variable(name, shape, initializer=tf.zeros_initializer(), dtype=tf.float32)
    return bias
    
    
def relu(x):
    return tf.nn.relu(x)
    
    
def lrelu(x, alpha=0.05):
    return tf.nn.leaky_relu(x, alpha)
    
    
def conv2d(x, shape, name, stride=[1, 1, 1, 1], pad='SAME', act='lrelu', alpha=0.05, use_bias=True):
    w_name = name + '_w'
    b_name = name + '_b'
    weight = weight_variable(shape, w_name)
    
    y = tf.nn.conv2d(x, weight, strides=stride, padding=pad)
    if use_bias is True:
        bias = bias_variable(shape[3], b_name)
        y = y + bias
    
    if act == 'relu':
        y = relu(y)
    elif act == 'lrelu':
        y = lrelu(y, alpha)

    #tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)

    return y

    
def group_conv2d(x, group, shape, block_index, group_index):
    group_size = shape[2] // group
    split_shape = [group_size, group_size, group_size, shape[2] - group_size * 3]
    group_feature = tf.split(x, split_shape, axis=3)
    layers = []
    g = 1
    for feature in group_feature:
        name = 'block' + str(block_index) + '_enhancement_group_conv' + str(group_index) + '_group' + str(g)
        g = g + 1
        layer = conv2d(
            feature, shape=[3, 3, shape[2] // group, shape[3] // group],
            act='lrelu', name=name)
        layers.append(layer)
    y = tf.concat(layers, axis=3)
    return y

    
def enhancement_unit(input_image, block_index):
    temp = conv2d(input_image, shape=[3, 3, 64, 48], name='block'+str(block_index)+'_enhancement_conv1')
    temp = group_conv2d(temp, group=4, shape=[3, 3, 48, 32], block_index=block_index, group_index=1)
    temp = conv2d(temp, shape=[3, 3, 32, 64], name='block'+str(block_index)+'_enhancement_conv3')
    
    temp1, temp2 = tf.split(temp, [16, 48], axis=3)#64//4=16
    
    temp2 = conv2d(temp2, shape=[3, 3, 48, 64], name='block'+str(block_index)+'_enhancement_conv4')
    temp2 = group_conv2d(temp2, group=4, shape=[3, 3, 64, 48], block_index=block_index, group_index=2)
    temp2 = conv2d(temp2, shape=[3, 3, 48, 80], name='block'+str(block_index)+'_enhancement_conv6')
    
    output_image = tf.concat([input_image, temp1], axis=3) + temp2

    return output_image
    

def compression_unit(input_image, block_index):
    output_image = conv2d(input_image, shape=[1, 1, 80, 64], name='block'+str(block_index)+'_compression_conv1')
    return output_image

    
def feature_extraction_block(input_image):
    temp = conv2d(input_image, shape=[3, 3, 1, 64], name='FBlock_conv1')
    output_image = conv2d(temp, shape=[3, 3, 64, 64], name='FBlock_conv2')
    return output_image
    
    
def information_distillation_block(input_image, block_index):
    temp = enhancement_unit(input_image, block_index)
    output_image = compression_unit(temp, block_index)
    return output_image


def reconstruction_block(input_image, scale):
    shape = tf.shape(input_image)
    output_shape = [shape[0], scale * shape[1], scale * shape[2], 1]
    stride = [1, scale, scale, 1]

    weight = weight_variable(shape=[17, 17, 1, 64], name='weight_reconstruction')
    bias = bias_variable(shape=output_shape[-1], name='bias_reconstruction')
    
    reconstruct_image = tf.nn.conv2d_transpose(input_image, weight, output_shape, stride, padding='SAME') + bias
    return reconstruct_image


def IDN(image, scale):
    temp = feature_extraction_block(image)
    
    for i in range(4):
        temp = information_distillation_block(temp, i)
    
    output = reconstruction_block(temp, scale)
    
    shape = tf.shape(image)[1:3]
    interpolated_low_resolution = tf.image.resize_images(
        image,
        (scale * shape[0], scale * shape[1]),
        method=tf.image.ResizeMethod.BICUBIC
    )
    output = output + interpolated_low_resolution
    return output