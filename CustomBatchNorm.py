import tensorflow as tf
from tensorflow.keras.layers import Layer 
from tensorflow.keras.layers import BatchNormalization
import numpy as np
from load_data import DataGenerator, DataLoader

class CustomBatchNormalization(Layer):
    def __init__(self, **kwargs):
        super(CustomBatchNormalization, self).__init__(**kwargs)
        self.batch_norm = BatchNormalization()

    def build(self, input_shape):
        super(CustomBatchNormalization, self).build(input_shape)

    def call(self, input_tensor):
        mask = tf.range(tf.shape(input_tensor)[-1]) % 2 != 0
        non_mask = tf.boolean_mask(mask, input_tensor, axis=-1)

        normalised_layers = self.batch_norm(non_mask)

        output = tf.where(mask, normalised_layers, input_tensor)

        return output


tensor = [0, 1, 2, 3]  # 1-D example
mask = np.array([True, False, True, False])
print(tf.boolean_mask(tensor, mask))


tensor = [[[0, 1], [0, 1], [0, 1]]]
mask = tf.range(tf.shape(tensor)[-1]) % 2 != 0
output = tf.where(mask, tensor, tensor)


print(mask)
print(output)


data_loader = DataLoader(time_period=[2004245, 2005100])
x_files, y_files = data_loader.get_files()
training_data = DataGenerator(x_files, y_files, lat_range=[-88, 0], lon_range=[-177.5, -2.5])
batch_X, batch_y = training_data[0]

print(batch_X.shape)

batch_norm = CustomBatchNormalization()
batch_norm.build(batch_X.shape)
print(batch_X[0][-10][-10])
batch_norm.call(batch_X)