# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Converts mydataset to TFRecords of TF-Example protos.

This moduel reads the files that make up the Flowers data and creates two TFRecord datasets: one for train and one for test. Each TFRecord dataset is comprised of a set of TF-Example protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

# The number of images in the validation set.
_NUM_VALIDATION = 500

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir, extra_dir, test_dir, ori_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.
    extra_dir: A directory containing a set of unlabeled images, but we can
      use it by semi-supervised approachs.
    test_dir: A directory containing a set of unlabeled images, as the final
      test images.

  Returns:
    A list of image file paths, relative to `dataset_dir`, a list of filenames 
    in extra_dir, a list of filenames in test_dir and the list of 
    subdirectories, representing class names.
  """
  dataset_root = os.path.join(dataset_dir, '')
  directories = []
  class_names = []
  for filename in os.listdir(dataset_root):
    path = os.path.join(dataset_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  extra_filenames = []
  test_filenames = []
  ori_filenames = []  

  if extra_dir:
    for filename in os.listdir(extra_dir):
      path = os.path.join(extra_dir, filename)
      extra_filenames.append(path)

  if test_dir:
    for filename in os.listdir(test_dir):
      path = os.path.join(test_dir, filename)
      test_filenames.append(path)

  if ori_dir:
    for filename in os.listdir(ori_dir):
      path = os.path.join(ori_dir, filename)
      ori_filenames.append(path)

  return photo_filenames, extra_filenames, test_filenames, ori_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'mydataset_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train', 'extra', 'test'
                 or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation', 'extra', 'test', 'ori']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d %s' % (
                i+1, len(filenames), shard_id, filenames[i]))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            if split_name in ['train', 'validation']:
              class_name = os.path.basename(os.path.dirname(filenames[i]))
              class_id = class_names_to_ids[class_name]
            elif split_name == 'extra':
              fid = int(os.path.basename(filenames[i]).split('.')[0])
              if fid <= 100:
                class_id = 9
              elif fid <= 395:
                class_id = 4
              elif fid <= 700:
                class_id = 8
              elif fid <= 1130:
                class_id = 7
              elif fid <= 1434:
                class_id = 1
              elif fid <= 1738:
                class_id = 6
              elif fid <= 2046:
                class_id = 0
              elif fid <= 2364:
                class_id = 10
              elif fid <= 2791:
                class_id = 11
              elif fid <= 3221:
                class_id = 5
              elif fid <= 66707:
                class_id = 3
              else:
                class_id = 2
            elif split_name == 'ori':
              ftype = os.path.basename(filenames[i]).split('_')[0]
              if ftype == 'n01613177':
                class_id = 9
              elif ftype == 'n01923025':
                class_id = 4
              elif ftype == 'n02278980':
                class_id = 8
              elif ftype == 'n03767203':
                class_id = 7
              elif ftype == 'n03877845':
                class_id = 1
              elif ftype == 'n04515003':
                class_id = 6
              elif ftype == 'n04583620':
                class_id = 0
              elif ftype == 'n07897438':
                class_id = 10
              elif ftype == 'n10247358':
                class_id = 11
              elif ftype == 'n11669921':
                class_id = 5
              else:
                class_id = 2
                raise ValueError(filenames[i] + ': Invalid type')
            else:
              class_id = 0  # Actually extra and test set are all unlabeled images

            example = dataset_utils.image_to_tfexample(
                image_data, 'jpg', height, width, class_id, filenames[i])
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, 'flower_photos')
  tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir, split_names):
  for split_name in split_names:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir, extra_dir=None, test_dir=None, ori_dir=None, split_name=None):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if not split_name:
    split_names = ['train', 'validation', 'extra', 'test']
  else:
    split_names = split_name.split(',')

  if _dataset_exists(dataset_dir, split_names):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  #dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
  photo_filenames, extra_filenames, test_filenames, ori_filenames, class_names = \
     _get_filenames_and_classes(dataset_dir, extra_dir, test_dir, ori_dir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  training_filenames = photo_filenames[_NUM_VALIDATION:]
  validation_filenames = photo_filenames[:_NUM_VALIDATION]

  # First, convert the training and validation sets.
  if 'train' in split_names:
    _convert_dataset('train', training_filenames, class_names_to_ids,
                   dataset_dir)
  if 'validation' in split_names:
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                   dataset_dir)

  # Convert extra set
  if extra_dir and 'extra' in split_names:
    _convert_dataset('extra', extra_filenames, class_names_to_ids,
                     dataset_dir)
  
  # Convert test set
  if test_dir and 'test' in split_names:
    _convert_dataset('test', test_filenames, class_names_to_ids, 
                     dataset_dir)

  # Convert ori set
  if 'ori' in split_names:
    _convert_dataset('ori', ori_filenames, class_names_to_ids, dataset_dir)  

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  #_clean_up_temporary_files(dataset_dir)
  print('\nFinished converting mydataset!')

