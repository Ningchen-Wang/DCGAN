import tensorflow as tf
import tensorflow.contrib.graph_editor as graph_editor

def leaky_relu(features, alpha=0.2):
  return tf.maximum(features, alpha * features)

_graph_replace = tf.contrib.graph_editor.graph_replace

def remove_original_op_attributes(graph):
  """Remove _original_op attribute from all operations in a graph."""
  for op in graph.get_operations():
    op._original_op = None
               
def graph_replace(*args, **kwargs):
  """Monkey patch graph_replace so that it works with TF 1.0"""
  remove_original_op_attributes(tf.get_default_graph())
  return _graph_replace(*args, **kwargs)
