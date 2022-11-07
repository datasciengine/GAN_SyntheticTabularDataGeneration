import tensorflow as tf 
from tensorflow import keras


class Generator(keras.Model):
    def __init__(self, out_shape):

        self.out_shape = out_shape
        super(Generator, self).__init__()
        self.dense_1 = keras.layers.Dense(128, input_dim=100, name="Dense")
        self.bn0 = keras.layers.BatchNormalization(name="BatchNorm0")

        self.dense_2 = keras.layers.Dense(256, name="Dense2")
        self.bn1 = keras.layers.BatchNormalization(name="BatchNorm1")

        self.dense_3 = keras.layers.Dense(512, name="Dense3")
        self.bn2 = keras.layers.BatchNormalization(name="BatchNorm2")

        self.dense_out = keras.layers.Dense(self.out_shape, activation="tanh")


    def call(self, x):
        x = self.dense_1(x)
        x = tf.nn.leaky_relu(self.bn0(x))
    
        x = self.dense_2(x)
        x = tf.nn.leaky_relu(self.bn1(x))
        
        x = self.dense_3(x)
        x = tf.nn.leaky_relu(self.bn2(x))
        
        x = self.dense_out(x)
        
        return x


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
        # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


    generator = Generator()
    noise = tf.random.normal((1, 100))
    generated_image = generator(noise)
    print(generated_image.shape)