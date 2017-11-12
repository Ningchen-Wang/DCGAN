import tensorflow as tf
import tensorflow.contrib.slim as slim
import pprint
import os

from datasets import dataset_factory

from dcgan import dcgan_generator, dcgan_discriminator
from train import dcgan_train_step
from tensorflow.python.training import optimizer
from collections import OrderedDict
from utils import graph_replace
#from keras.optimizers import Adam

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 128, 'The size of minibatch when training [128]')
flags.DEFINE_float('learning_rate', 2e-4, 'Learning rate of optimizer [2e-4]')
flags.DEFINE_string('optimizer', 'Adam', 'Optimizer used when training [Adam]')
flags.DEFINE_string('dataset_name', 'mnist', 'Image dataset used when trainging [mnist]')
flags.DEFINE_string('split_name', 'train', 'Split name of dataset [train]')
flags.DEFINE_string('dataset_dir', './data/mnist/', 'Path to dataset directory [./data/mnist]')
flags.DEFINE_string('checkpoint_path', None, 'Path to checkpoint path [None]')
flags.DEFINE_string('train_dir', './train', 'Path to save new training result [./train]')
flags.DEFINE_integer('max_step', 1000, 'Maximum training steps [1000]')
flags.DEFINE_integer('z_dim', 100, 'z-dim for generator [100]')
flags.DEFINE_float('beta1', 0.5, 'Beta1 for Adam optimizer [0.5]')
flags.DEFINE_float('beta2', 0.999, 'Beta2 for Adam optimizer [0.999]')
flags.DEFINE_float('epsilon', 1e-8, 'Epsilon for Adam optimizer [1e-8]')
flags.DEFINE_integer('log_every_n_steps', 10, 'Log every n training steps [10]')
flags.DEFINE_integer('save_interval_secs', 600, 'How often, in seconds, to save the model checkpoint [600]')
flags.DEFINE_integer('save_summaries_secs', 10, 'How often, in seconds, to save the summary [10]')
flags.DEFINE_integer('sample_n', 16, 'How many images the network will produce in a sample process [16]')
flags.DEFINE_integer('unrolled_step', 0, 'Unrolled step in surrogate loss for generator [0]')

def extract_update_dict(update_ops):
  """Extract the map between tensor and its updated version in update_ops"""
  update_map = OrderedDict()
  name_to_var = {v.name: v for v in tf.global_variables()}
  for u in update_ops:
    var_name = u.op.inputs[0].name
    var = name_to_var[var_name]
    value = u.op.inputs[1]
    if u.op.type == 'Assign':
      update_map[var.value()] = value
    elif u.op.type == 'AssignAdd':
      update_map[var.value()] = value + var
    else:
      raise ValueError('Undefined unroll update type %s', u.op.type)
  return update_map

def main(*args):
  FLAGS = flags.FLAGS

  pp = pprint.PrettyPrinter(indent=4)
  print('Running flags:')
  pp.pprint(FLAGS.__dict__['__flags'])
  tf.logging.set_verbosity('INFO') 
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Graph().as_default():
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.split_name, FLAGS.dataset_dir), 
                                        common_queue_capacity=2*FLAGS.batch_size,
                                        common_queue_min=FLAGS.batch_size)
    [image] = provider.get(['image'])
    image = tf.to_float(image)
    image = tf.subtract(tf.divide(image, 127.5), 1)
    z = tf.random_uniform(shape=([FLAGS.z_dim]), minval=-1, maxval=1, name='z')
    label_true = tf.random_uniform(shape=([]), minval=0.7, maxval=1.2, name='label_t')
    label_false = tf.random_uniform(shape=([]), minval=0, maxval=0.3, name='label_f')
    sampler_z = tf.random_uniform(shape=([FLAGS.batch_size, FLAGS.z_dim]), minval=-1, maxval=1, name='sampler_z')
    [image, z, label_true, label_false] = tf.train.batch([image, z, label_true, label_false], batch_size=FLAGS.batch_size, capacity=2*FLAGS.batch_size)
    generator_result = dcgan_generator(z, 'Generator', reuse=False, output_height=28, fc1_c=1024, grayscale=True)
    sampler_result = dcgan_generator(sampler_z, 'Generator', reuse=True, output_height=28, fc1_c=1024, grayscale=True)
    discriminator_g, g_logits = dcgan_discriminator(generator_result, 'Discriminator', reuse=False, conv2d1_c=128, grayscale=True)
    discriminator_d, d_logits = dcgan_discriminator(image, 'Discriminator', reuse=True, conv2d1_c=128, grayscale=True)
    d_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=label_false, logits=g_logits) + \
             tf.losses.sigmoid_cross_entropy(multi_class_labels=label_true, logits=d_logits)
    standard_g_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=label_true, logits=g_logits)
    if FLAGS.optimizer == 'Adam':
      g_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                           beta1=FLAGS.beta1,
                                           beta2=FLAGS.beta2,
                                           epsilon=FLAGS.epsilon,
                                           name='g_adam')
      d_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                           beta1=FLAGS.beta1,
                                           beta2=FLAGS.beta2,
                                           epsilon=FLAGS.epsilon,
                                           name='d_adam')
      #unrolled_optimizer = Adam(lr=FLAGS.learning_rate,
      #                                           beta_1=FLAGS.beta1,
      #                                           beta_2=FLAGS.beta2,
      #                                           epsilon=FLAGS.epsilon)

    elif FLAGS.optimizer == 'SGD':
      g_optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate,
                                                      name='g_sgd')
      d_optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate,
                                                      name='d_sgd')

    var_g = slim.get_variables(scope='Generator', collection=tf.GraphKeys.TRAINABLE_VARIABLES)
    var_d = slim.get_variables(scope='Discriminator', collection=tf.GraphKeys.TRAINABLE_VARIABLES)
    #update_ops = unrolled_optimizer.get_updates(var_d, [], d_loss)
    #update_map = extract_update_dict(update_ops)
    #current_update_map = update_map
    #pp.pprint(current_update_map)
    current_update_map = OrderedDict()
    for i in xrange(FLAGS.unrolled_step):
      grads_d = list(zip(tf.gradients(standard_g_loss, var_d), var_d))
      update_map = OrderedDict()
      for g, v in grads_d:
        update_map[v.value()] = v + g * FLAGS.learning_rate
      current_update_map = graph_replace(update_map, update_map)
      pp.pprint(current_update_map)
    if FLAGS.unrolled_step != 0:
      unrolled_loss = graph_replace(standard_g_loss, current_update_map)
      g_loss = unrolled_loss
    else:
      g_loss = standard_g_loss
    generator_global_step = slim.variable("generator_global_step", 
                                          shape=[], 
                                          dtype=tf.int64, 
                                          initializer=tf.zeros_initializer,
                                          trainable=False)
    discriminator_global_step = slim.variable("discriminator_global_step",
                                              shape=[],
                                              dtype=tf.int64,
                                              initializer=tf.zeros_initializer,
                                              trainable=False)
    global_step = slim.get_or_create_global_step()
    with tf.name_scope('train_step'):
      train_step_kwargs = {}
      train_step_kwargs['g'] = generator_global_step
      train_step_kwargs['d'] = discriminator_global_step
      if FLAGS.max_step:
        train_step_kwargs['should_stop'] = tf.greater_equal(global_step, FLAGS.max_step)
      else:
        train_step_kwargs['should_stop'] = tf.constant(False)
      train_step_kwargs['should_log'] = tf.equal(tf.mod(global_step, FLAGS.log_every_n_steps), 0)
    train_op_d = slim.learning.create_train_op(d_loss, d_optimizer, variables_to_train=var_d, global_step=discriminator_global_step)
    train_op_g = slim.learning.create_train_op(g_loss, g_optimizer, variables_to_train=var_g, global_step=generator_global_step)
    train_op_s = tf.assign_add(global_step, 1)
    train_op = [train_op_g, train_op_d, train_op_s]
    def group_image(images):
      sampler_result = tf.split(images, FLAGS.batch_size // 16, axis=0)
      group_sample = []
      for sample in sampler_result:
        unstack_sample = tf.unstack(sample, num=16, axis=0)
        group_sample.append(tf.concat([tf.concat([unstack_sample[i*4+j] for j in range(4)], axis=1) for i in range(4)], axis=0))
      sampler_result = tf.stack(group_sample, axis=0)
      return sampler_result
    sampler_result = group_image(sampler_result)
    train_data = group_image(image)
    tf.summary.image('sampler_z', sampler_result, max_outputs=16)
    tf.summary.image('train_data', train_data, max_outputs=16)
    tf.summary.scalar('loss/g_loss', g_loss)
    tf.summary.scalar('loss/d_loss', d_loss)
    tf.summary.scalar('loss/standard_g_loss', standard_g_loss)
    tf.summary.scalar('loss/total_loss', g_loss + d_loss)
    tf.summary.scalar('loss/standard_total_loss', standard_g_loss + d_loss)
    if FLAGS.unrolled_step != 0:
      tf.summary.scalar('loss/unrolled_loss', unrolled_loss)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())
      if FLAGS.checkpoint_path:
        if not tf.train.checkpoint_exists(FLAGS.checkpoint_path):
          raise ValueError('Checkpoint not exist in path ', FLAGS.checkpoint_path)
        else:
          restore_vars = slim.get_variables_to_restore(exclude_patterns=split(FLAGS.exclude_scope, ','))
          sess.run(slim.assign_from_checkpoint(FLAGS.checkpoint_path, restore_vars, ignore_missing_vars=False))
      slim.learning.train(train_op,
                          FLAGS.train_dir,
                          global_step=global_step,
                          train_step_fn=dcgan_train_step,
                          train_step_kwargs=train_step_kwargs,
                          saver=saver,
                          save_interval_secs=FLAGS.save_interval_secs,
                          save_summaries_secs=FLAGS.save_summaries_secs)
  return

if __name__ == '__main__':
    tf.app.run(main=main)

