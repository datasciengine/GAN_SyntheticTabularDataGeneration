import tensorflow as tf 
from tensorflow import keras


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dense_1 = keras.layers.Dense(512, input_shape=(14, 1))

        self.dense_2 = keras.layers.Dense(256)

        # self.dense_3 = keras.layers.Dense(128)

        self.dense_out = keras.layers.Dense(1, activation='sigmoid')


        
    def call(self, x, training=False):
        x = tf.nn.leaky_relu(self.dense_1(x))
        if training:
            x = tf.nn.dropout(x, 0.3)
            
        x = tf.nn.relu(self.dense_2(x))
        if training:
            x = tf.nn.dropout(x, 0.3)
        
        return self.dense_out(x)
 