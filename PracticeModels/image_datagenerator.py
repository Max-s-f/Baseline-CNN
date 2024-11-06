import numpy as np
import keras 
import tensorflow as tf

class DataGenerator(keras.utils.Sequence):

    def __init__(self, images, batch_size=32, shuffle=True, data_aug=True):
        
        _, H, W, C = np.shape(images)
        self.dim = (H, W, C)
        self.batch_size = batch_size
        self.images = images
        self.shuffle = shuffle
        self.num_samples = len(self.images)
        self.data_aug = data_aug
        self.on_epoch_end()


    def on_epoch_end(self):
        self.indexes = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def noise(self, array):
        """Adds random noise to each image in the supplied array."""
        noise_factor = 0.3
        noisy_array = array + noise_factor * np.random.normal(
            loc=0.0, scale=1.0, size=array.shape
        )
        return np.clip(noisy_array, 0.0, 1.0)


    def __data_generation(self, list_IDs_temp):
        H, W, C = self.dim
        X = np.empty((self.batch_size, H, W, C))
        if self.data_aug:
            X_noise = np.empty((self.batch_size, H, W, C))

        for i, ID in enumerate(list_IDs_temp):
            X[i] = self.images[ID]
            if self.data_aug:
                # Some random augmentation
                X_noise[i] = self.noise(X[i])        if self.data_aug:
            return X_noise, X
        else:
            return X, X


    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))


    def __getitem__(self, index):
        list_IDs_temp = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        X, y = self.__data_generation(list_IDs_temp)
        return X, y