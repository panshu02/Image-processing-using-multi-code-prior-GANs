import tensorflow as tf

class AdaIN(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon

    def call(self, inputs):
        x, style = inputs
        x_mean, x_var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_std = tf.sqrt(x_var + self.epsilon)

        style_mean, style_var = tf.nn.moments(style, axes=[1, 2], keepdims=True)
        style_std = tf.sqrt(style_var + self.epsilon)

        normalized = (x - x_mean) / x_std
        stylized = normalized * style_std + style_mean

        return stylized