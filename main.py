import tensorflow as tf
import tensorflow.contrib.slim as slim
import pprint
import os

from dcgan import dcgan_generator, dcgan_discriminator
from train import dcgan_train_step

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 128, 'The size of minibatch when training [128]')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate of optimizer [1e-3]')
flags.DEFINE_string('optimizer', 'Adam', 'Optimizer used when training [Adam]')
flags.DEFINE_string('dataset_name', 'mnist', 'Image dataset used when trainging [mnist]')
flags.DEFINE_string('dataset_dir', '', 'Path to dataset directory []')
flags.DEFINE_string('checkpoint_path', None, 'Path to checkpoint path [None]')
flags.DEFIEN_string('train_dir', '', 'Path to save new training result')

def main(*args):
    FLAGS = flags.FLAGS
    FLAGS.dataset_dir = os.path.join(os.getcwd(), FLAGS.dataset_dir)
    FLAGS.checkpoint_dir = os.path.join(os.getcwd(), FLAGS.checkpoint_dir)

    pp = pprint.PrettyPrinter(indent=4)
    print('Running flags:')
    pp.pprint(FLAGS.__dict__['__flags'])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    z = tf.placeholder(tf.float32, shape=(z_dim), name='z')
    t = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 64, 64, 1), name='t')
    generator_result = dcgan_generator(z, 'Generator', reuse=False, output_height=64, fc1_c=1024, greyscale=True)
    discriminator_g = dcgan_discriminator(generator_result, 'DCGAN', reuse=False, conv2d1_c=128, greyscale=True)
    discriminator_t = dcgan_discriminator(t, 'Discriminator', reuse=True, conv2d1_c=128, greyscale=True)
    g_loss = tf.losses.sigmoid_cross_entropy=(multi_class_labels=tf.ones(FLAGS.batch_size), logits=discriminator_g)
    d_loss = tf.losses.sigmoid_cross_entropy=(multi_class_labels=tf.zeros(FLAGS.batch_size), logits=discriminator_g) + \
             tf.losses.sigmoid_cross_entropy=(multi_class_labels=tf.ones(FLAGS.batch_szie), logits=discriminator_t)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                         beta1=FLAGS.beta1,
                                         beta2=FLAGS.beta2,
                                         epsilon=FLAGS.epsilon,
                                         name='g_adam')
    d_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                         beta1=FLAGS.beta1,
                                         beta2=FLAGS.beta2,
                                         epsilon=FLAGS.epsilon,
                                         name='d-adam')
    saver = tf.train.Saver()
    provider = slim.DatasetDataProvider(FLAGS.dataset_name, 
                                        common_queue_capacity=2*FLAGS.batch_size,
                                        common_queue_min=FLAGS.batch_size)
    var_g = slim.get_variables(scope=tf.variable_scope('Generator'))
    var_d = slim.get_variables(scope=tf.variable_scope('Discriminator'))
    generator_global_step = slim.variable("generator_global_step", 
                                          shape=[], 
                                          dtype=tf.uint64, 
                                          initializer=tf.zeros_initializer,
                                          trainable=False)
    discriminator_global_step = slim.variable("discriminator_global_step",
                                              shape=[],
                                              dtype=tf.uint64,
                                              initializer=tf.zeros.initializer,
                                              traibable=False)
    with tf.name_scope('train_step'):
      train_step_kwargs = {}
      train_step_kwargs['g'] = generator_global_step
      train_step_kwargs['d'] = discriminator_global_step
      if FLAGS.max_step:
        train_step_kwargs['should_stop'] = tf.greater_equal(global_step, FLAGS.max_step)
      else:
        train_step_kwargs['should_stop'] = tf.constant(False)
      train_step_kwargs['should_log'] = tf.equals(tf.mod(global_step, FLAGS.log_every_n_steps), 0)
    train_op_g = slim.learning.create_train_op(g_loss, g_optimizer, variables_to_train=var_g, global_step=generator_global_step)
    train_op_d = slim.learning.create_train_op(d_loss, d_optimizer, variables_to_train=var_d, global_step=discriminator_global_step)
    train_op = [train_op_g, train_op_d]
   with tf.Session(config=config) as sess:
      if FLAGS.checkpoint_path:
        if not tf.train.checkpoint_exists(FLAGS.checkpoint_path):
          raise ValueError('Checkpoint not exist in path ', FLAGS.checkpoint_path)
        else:
          restore_vars = slim.get_variables_to_restore(exclude_patterns=split(FLAGS.exclude_scope, ','))
          sess.run(slim.assign_from_checkpoint(FLAGS.checkpoint_path, restore_vars, ignore_missing_vars=False)
      else:
        sess.run(tf.global_variables_initlizer())
      slim.learning.train(train_op, FLAGS.train_dir, train_step_fn=dcgan_train_step, train_step_kwargs=global_steps)
   return

if __name__ == '__main__':
    tf.app.run(main=main)

