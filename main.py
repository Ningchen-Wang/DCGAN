import tensorflow as tf
import pprint
import os

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 128, 'The size of minibatch when training [128]')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate of optimizer [1e-3]')
flags.DEFINE_string('optimizer', 'Adam', 'Optimizer used when training [Adam]')
flags.DEFINE_string('dataset_name', 'mnist', 'Image dataset used when trainging [mnist]')
flags.DEFINE_string('dataset_dir', '', 'Path to dataset directory []')
flags.DEFINE_string('checkpoint_dir', '', 'Path to checkpoint directory []')

def main(*args):
    FLAGS = flags.FLAGS
    FLAGS.dataset_dir = os.path.join(os.getcwd(), FLAGS.dataset_dir)
    FLAGS.checkpoint_dir = os.path.join(os.getcwd(), FLAGS.checkpoint_dir)

    pp = pprint.PrettyPrinter(indent=4)
    print('Running flags:')
    pp.pprint(FLAGS.__dict__['__flags'])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        dcgan(sess, FLAGS)
    return

if __name__ == '__main__':
    tf.app.run(main=main)

