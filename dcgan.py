from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import leaky_relu

def conv2d_transpose(inputs, output_shape, kernel_size, stride, padding, activation_fn, normalizer_fn, scope):
  with tf.variable_scope(scope, 'conv2d_transpose'):
    filter_ = tf.get_variable('filter', [kernel_size, kernel_size, output_shape[3], inputs.shape[3]],
                              trainable=True, 
                              collections=[tf.GraphKeys.TRAINABLE_VARIABLES, 
                                           tf.GraphKeys.MODEL_VARIABLES,
                                           tf.GraphKeys.GLOBAL_VARIABLES],
                              dtype=tf.float32)
    conv_t = tf.nn.conv2d_transpose(value=inputs,
                                    filter=filter_,
                                    output_shape=output_shape,
                                    strides=[1, stride, stride, 1],
                                    padding=padding,
                                    name=scope)
    if normalizer_fn:
      conv_t = normalizer_fn(conv_t)
    if activation_fn:
      conv_t = activation_fn(conv_t)
   
  return conv_t

def dcgan_generator(inputs, scope, reuse=None, output_height=64, 
                    output_width=None, fc1_c=1024, grayscale=False):
  if not output_width:
    output_width = output_height

  with tf.variable_scope(scope, 'Generator', [inputs], reuse=reuse):
    fc1_h = int(math.ceil(output_height / 16.0))
    fc1_w = int(math.ceil(output_width / 16.0))
    deconv2d2_h = int(math.ceil(output_height / 8.0))
    deconv2d2_w = int(math.ceil(output_width / 8.0))
    deconv2d3_h = int(math.ceil(output_height / 4.0))
    deconv2d3_w = int(math.ceil(output_width / 4.0))
    deconv2d4_h = int(math.ceil(output_height / 2.0))
    deconv2d4_w = int(math.ceil(output_width / 2.0))
    batch_size = inputs.shape[0].value

    if fc1_h == 0 or fc1_w == 0:
      raise ValueError('Now the code should not reach here fc1_h and fc1_w should not be 0')
    if fc1_c // 8 == 0:
      raise ValueError('The channel of Tensor fc1 is too small, at least 8')
    if grayscale:
      output_c = 1
    else:
      output_c = 3
    fc1 = slim.fully_connected(inputs=inputs, 
                               num_outputs=fc1_h*fc1_w*fc1_c,
                               activation_fn=None,
                               biases_initializer=tf.zeros_initializer,
                               biases_regularizer=slim.l2_regularizer,
                               scope='fc1')
    fc1_reshape = tf.reshape(fc1, [-1, fc1_h, fc1_w, fc1_c])
    bn1 = slim.batch_norm(inputs=fc1_reshape, activation_fn=tf.nn.relu, scope='bn1')
    deconv2d2 = conv2d_transpose(inputs=bn1,
                                 output_shape=[batch_size, deconv2d2_h, deconv2d2_w, fc1_c // 2],
                                 kernel_size=5,
                                 stride=2,
                                 padding='SAME',
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=slim.batch_norm,
                                 scope='deconv2d2')
    deconv2d3 = conv2d_transpose(inputs=deconv2d2,
                                 output_shape=[batch_size, deconv2d3_h, deconv2d3_w, fc1_c // 4],
                                 kernel_size=5,
                                 stride=2,
                                 padding='SAME',
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=slim.batch_norm,
                                 scope='deconv2d3')
    deconv2d4 = conv2d_transpose(inputs=deconv2d3,
                                 output_shape=[batch_size, deconv2d4_h, deconv2d4_w, fc1_c // 8],
                                 kernel_size=5,
                                 stride=2,
                                 padding='SAME',
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=slim.batch_norm,
                                 scope='deconv2d4')
    deconv2d5 = conv2d_transpose(inputs=deconv2d4,
                                 output_shape=[batch_size, output_height, output_width, output_c],
                                 kernel_size=5,
                                 stride=2,
                                 padding='SAME',
                                 activation_fn=tf.tanh,
                                 normalizer_fn=None,
                                 scope='deconv2d5')
  
    return deconv2d5

def dcgan_discriminator(inputs, scope, reuse=None, conv2d1_c=128, 
                        grayscale=False):
  if grayscale:
    if inputs.shape.ndims == 3:
      inputs = tf.expand_dims(inputs, -1)
    elif inputs.shape.ndims != 4 or inputs.shape[3] != 1:
      raise ValueError('Inputs shape is invalid for grayscale images, shape is ', inputs.shape.dims)
  else:
    if inputs.shape.dims != 4 or inputs.shape[3] != 3:
      raise ValueError('Inputs shape is invalid for normal images, shape is ', inputs.shape)

  with tf.variable_scope(scope, 'Discriminator', [inputs], reuse=reuse):
    conv2d1 = slim.conv2d(inputs=inputs,
                          num_outputs=conv2d1_c,
                          kernel_size=5,
                          stride=2,
                          padding='SAME',
                          activation_fn=leaky_relu,
                          normalizer_fn=None,
                          scope='conv2d1')
    conv2d2 = slim.conv2d(inputs=conv2d1,
                          num_outputs=conv2d1_c * 2,
                          kernel_size=5,
                          stride=2,
                          padding='SAME',
                          activation_fn=leaky_relu,
                          normalizer_fn=slim.batch_norm,
                          scope='conv2d2')

    conv2d3 = slim.conv2d(inputs=conv2d2,
                          num_outputs=conv2d1_c * 4,
                          kernel_size=5,
                          stride=2,
                          padding='SAME',
                          activation_fn=leaky_relu,
                          normalizer_fn=slim.batch_norm,
                          scope='conv2d3')

    conv2d4 = slim.conv2d(inputs=conv2d3,
                          num_outputs=conv2d1_c * 8,
                          kernel_size=5,
                          stride=2,
                          padding='SAME',
                          activation_fn=leaky_relu,
                          normalizer_fn=slim.batch_norm,
                          scope='conv2d4')
    flatten5 = slim.flatten(inputs=conv2d4, scope='flatten5')
    fc5 = slim.fully_connected(inputs=flatten5,
                               num_outputs=1,
                               biases_initializer=tf.zeros_initializer,
                               biases_regularizer=slim.l2_regularizer,
                               activation_fn=None,
                               scope='fc5')
    fc5 = tf.squeeze(fc5)
    tf.summary.histogram('conv2d1', conv2d1)
    tf.summary.histogram('conv2d2', conv2d2)
    tf.summary.histogram('conv2d3', conv2d3)
    tf.summary.histogram('conv2d4', conv2d4)
    tf.summary.histogram('fc5', fc5)
    tf.summary.histogram('fc5/weights', tf.get_default_graph().get_tensor_by_name('Discriminator/fc5/weights:0'))
    return tf.nn.sigmoid(fc5), fc5

