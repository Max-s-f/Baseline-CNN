import tensorflow as tf

class LoggingCallback(tf.keras.callbacks.Callback):

    prev_weights = []

    def __init__(self, model, **kwargs):
        super(LoggingCallback, self).__init__(**kwargs)
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        curr_weights = self.model.layers[2].get_weights()[0][0][0]
        if len(self.prev_weights) == 0:
            self.prev_weights = curr_weights
        else:
            print(curr_weights == self.prev_weights)
            self.prev_weights = curr_weights
        return super().on_epoch_end(epoch, logs)
