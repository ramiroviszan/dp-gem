import tensorflow as tf
from tensorflow.keras.layers import Layer

class Norm2Clipping(Layer):
    def __init__(self, **kwargs):
        super(Norm2Clipping, self).__init__ (**kwargs)
        
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def get_config(self):
        return super(Norm2Clipping, self).get_config()


class Norm1Clipping(Layer):
    def __init__(self, **kwargs):
        super(Norm1Clipping, self).__init__ (**kwargs)
        
    def call(self, inputs):
        norm = tf.reshape(tf.norm(inputs, ord=1, axis=1), [tf.shape(inputs)[0], 1])
        return tf.divide(inputs, tf.maximum(norm, 1))

    def get_config(self):
        return super(Norm1Clipping, self).get_config()
    