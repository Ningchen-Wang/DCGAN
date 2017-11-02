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

# The original code is tensorflow.contrib.slim.learning.train_step
# To make train step a take turn process for generator and discriminator in
# GAN-like networks

import time

import tensorflow as tf
import tensorflow.contrib.slim as slim

def dcgan_train_step(sess, train_ops, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.

  Args:
    sess: The current session.
    train_ops: An `Operation` list that evaluates the gradients and returns the
      total loss for generator and for discriminator.
    global_step: A `Tensor` list representing the global training step of
      generator and discriminator respectively.
    train_step_kwargs: A dictionary of keyword arguments.

  Returns:
    The generator loss, discriminator loss and a boolean indicating whether or 
    not to stop training.

  Raises:
    ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
  """
  start_time = time.time()
  assert(slim.get_global_step() == global_step)
  trace_run_options = None
  run_metadata = None
  if 'should_trace' in train_step_kwargs:
    if 'logdir' not in train_step_kwargs:
      raise ValueError('logdir must be present in train_step_kwargs when '
                       'should_trace is present')
    if sess.run(train_step_kwargs['should_trace']):
      trace_run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()

  generator_train_op = train_ops[0]
  discriminator_train_op = train_ops[1]
  global_step_train_op = train_ops[2]
  generator_global_step = train_step_kwargs['g']
  discriminator_global_step = train_step_kwargs['d']

  # Generator step 
  generator_loss, generator_global_step = sess.run([generator_train_op, generator_global_step],
                                            options=trace_run_options,
                                            run_metadata=run_metadata)
  # Discriminator step
  discriminator_loss, discriminator_global_step = sess.run([discriminator_train_op, discriminator_global_step],
                                                options=trace_run_options,
                                                run_metadata=run_metadata)
  # Increase global_step
  np_global_step = sess.run(global_step_train_op)

  time_elapsed = time.time() - start_time
  total_loss = generator_loss + discriminator_loss

  if run_metadata is not None:
    tl = timeline.Timeline(run_metadata.step_stats)
    trace = tl.generate_chrome_trace_format()
    trace_filename = os.path.join(train_step_kwargs['logdir'],
                                  'tf_trace-%d.json' % np_global_step)
    tf.logging.info('Writing trace to %s', trace_filename)
    file_io.write_string_to_file(trace_filename, trace)
    if 'summary_writer' in train_step_kwargs:
      train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                           'run_metadata-%d' %
                                                           np_global_step)

  if 'should_log' in train_step_kwargs:
    if sess.run(train_step_kwargs['should_log']):
      tf.logging.info('global step %d: loss = %.4f g_loss = %.4f d_loss = %.4f (%.3f sec/step)',
                     np_global_step, total_loss, generator_loss, discriminator_loss, time_elapsed)

  # TODO(nsilberman): figure out why we can't put this into sess.run. The
  # issue right now is that the stop check depends on the global step. The
  # increment of global step often happens via the train op, which used
  # created using optimizer.apply_gradients.
  #
  # Since running `train_op` causes the global step to be incremented, one
  # would expected that using a control dependency would allow the
  # should_stop check to be run in the same session.run call:
  #
  #   with ops.control_dependencies([train_op]):
  #     should_stop_op = ...
  #
  # However, this actually seems not to work on certain platforms.
  if 'should_stop' in train_step_kwargs:
    should_stop = sess.run(train_step_kwargs['should_stop'])
  else:
    should_stop = False

  return total_loss, should_stop
