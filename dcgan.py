from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import leaky_relu

def dcgan_generator(inputs, scope, reuse=None, output_height=64, 
                    output_width=None, fc1_c=1024, greyscale=False):
  if not output_width:
    output_width = output_height

  with tf.variable_scope(scope, 'Generator', [inputs], reuse=reuse):
    fc1_h = output_height // 16
    fc1_w = output_width // 16
    if fc1_h == 0 or fc1_w == 0:
      raise ValueError('The output of generator is too small, at least 16*16')
    if fc1_c // 8 == 0:
      raise ValueError('The channel of Tensor fc1 is too small, at least 8')
    if greyscale:
      output_c = 1
    else:
      output_c = 3
    fc1 = slim.fully_connected(inputs=inputs, 
                               num_outputs=fc1_h*fc1_w*fc1_c,
                               activation_fn=None,
                               scope='fc1')
    fc1_reshape = tf.reshape(fc1, [-1, fc1_h, fc1_w, fc1_c])
    bn1 = slim.batch_norm(inputs=fc1_reshape, activation_fn=tf.nn.relu, scope='bn1')
    deconv2d2 = slim.conv2d_transpose(inputs=bn1,
                                      num_outputs=fc1_c//2,
                                      kernel_size=5,
                                      stride=2,
                                      padding='SAME',
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=slim.batch_norm,
                                      scope='deconv2d2')
    deconv2d3 = slim.conv2d_transpose(inputs=deconv2,
                                      num_outputs=fc1_c//4,
                                      kernel_size=5,
                                      stride=2,
                                      padding='SAME',
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=slim.batch_norm,
                                      scope='deconv2d3')

    deconv2d4 = slim.conv2d_transpose(inputs=deconv3,
                                      num_outputs=fc1_c//8,
                                      kernel_size=5,
                                      stride=2,
                                      padding='SAME',
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=slim.batch_norm,
                                      scope='deconv2d4')
    deconv2d5 = slim.conv2d_transpose(inputs=deconv4,
                                      num_outputs=output_c,
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
    if inputs.shape.dims == 3:
      inputs = tf.expand_dims(inputs, -1)
    elif inputs.shape.dims != 4 or inputs.shape[3] != 1:
      raise ValueError('Inputs shape is invalid for grayscale images, shape is ', inputs.shape)
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
    fc5 = slim.fully_connected(inputs=conv2d4,
                               num_outputs=1,
                               activation_fn=tf.sigmoid,
                               scope='fc5')
  return fc5

