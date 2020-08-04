import tensorflow as tf
from tensorflow.keras.layers import Layer

class Norm2Clipping(Layer):
    def __init__(self):
        super(Norm2Clipping, self).__init__()
        
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


class Norm1Clipping(Layer):
    def __init__(self):
        super(Norm1Clipping, self).__init__()
        
    def call(self, inputs):
        norm = tf.norm(inputs, ord=1, axis=1)
        return tf.divide(inputs, tf.reshape(norm, [tf.shape(inputs)[0], 1]))
 