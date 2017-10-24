import tensorflow as tf

def leaky_relu(features, alpha=0.2):
    return tf.maximum(features, alpha * features)
