import tensorflow as tf
import tensorflow.contrib.slim as slim
import pprint
import os

from datasets import dataset_factory

from dcgan import dcgan_generator, dcgan_discriminator
from train import dcgan_train_step

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
flags.DEFINE_integer('save_interval_secs', 600, 'How often, in seconds, to save the model checkpoint')

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
    [image, z] = tf.train.batch([image, z], batch_size=FLAGS.batch_size, capacity=2*FLAGS.batch_size)
    generator_result = dcgan_generator(z, 'Generator', reuse=False, output_height=28, fc1_c=1024, grayscale=True)
    discriminator_g = dcgan_discriminator(generator_result, 'Discriminator', reuse=False, conv2d1_c=128, grayscale=True)
    discriminator_t = dcgan_discriminator(image, 'Discriminator', reuse=True, conv2d1_c=128, grayscale=True)
    g_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones(FLAGS.batch_size), logits=discriminator_g)
    d_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros(FLAGS.batch_size), logits=discriminator_g) + \
             tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones(FLAGS.batch_size), logits=discriminator_t)
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
    elif FLAGS.optimizer == 'SGD':
      g_optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate,
                                                      name='g_sgd')
      d_optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate,
                                                      name='d_sgd')

    var_g = slim.get_variables(scope='Generator', collection=tf.GraphKeys.TRAINABLE_VARIABLES)
    var_d = slim.get_variables(scope='Discriminator', collection=tf.GraphKeys.TRAINABLE_VARIABLES)
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
    saver = tf.train.Saver()
    with tf.name_scope('train_step'):
      train_step_kwargs = {}
      train_step_kwargs['g'] = generator_global_step
      train_step_kwargs['d'] = discriminator_global_step
      if FLAGS.max_step:
        train_step_kwargs['should_stop'] = tf.greater_equal(global_step, FLAGS.max_step)
      else:
        train_step_kwargs['should_stop'] = tf.constant(False)
      train_step_kwargs['should_log'] = tf.equal(tf.mod(global_step, FLAGS.log_every_n_steps), 0)
    train_op_g = slim.learning.create_train_op(g_loss, g_optimizer, variables_to_train=var_g, global_step=generator_global_step)
    train_op_d = slim.learning.create_train_op(d_loss, d_optimizer, variables_to_train=var_d, global_step=discriminator_global_step)
    train_op_s = tf.assign_add(global_step, 1)
    train_op = [train_op_g, train_op_d, train_op_s]
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
                          save_interval_secs=FLAGS.save_interval_secs)
  return

if __name__ == '__main__':
    tf.app.run(main=main)

