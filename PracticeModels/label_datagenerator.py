import numpy as np
import keras
import tensorflow as tf

class DataGenerator(keras.utils.Sequence):
    def __init__(self, images, labels, batch_size = 32, shuffle = True, data_aug=True):
        
        _, H, W, C = np.shape(images)
        self.dim = (H,W,C)
        self.batch_size = batch_size
        self.labels = labels
        self.images = images
        self.shuffle = shuffle
        self.num_samples = len(self.images)
        self.data_aug = data_aug
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        H,W,C = self.dim
        X = np.empty((self.batch_size,H,W,C))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            X[i] = self.images[ID]
            if self.data_aug:
                # Gray scale
                X[i] = tf.image.rgb_to_grayscale(X[i])

            y[i] = self.labels[ID]

        #return X, keras.utils.to_categorical(y, num_classes = self.n_classes)
        return X, y


    def __len__(self):
        'Denotes no. batches p epoch'
        return int(np.floor(len(self.images) / self.batch_size))


    def __getitem__(self, index):
        list_IDs_temp = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y




    # train_images, train_labels, test_images, text_labels = load_cifar10()

    # train_data = DataGenerator(images=train_images, labels=train_labels)
    # valid_data = DataGenerator(images=valid_image, labels=valid_labels, shuffle=False, data_aug=False)

    # test_data = DataGenerator(images=test_images, labels=test_label, shuffle=False, data_aug=False)



    # #Model created 



    # model.compile(......, run_eagerly=True)

    """
    Look at callback functions
    in example 4-6a on Lech's tf intro it shows how to do it
    """
    # model.train(train_data, validation = valid_data, #callback=)



