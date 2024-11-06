from keras.losses import Loss, MSE
from keras.utils.losses_utils import ReductionV2
import tensorflow as tf


# pass K as parameter into initialisation
class MeanSquaredErrorX(Loss):

    # W and b normalisation factors
    def __init__(self, K):
        super().__init__()
        self.K = K


    def call(self, y_true, y_pred):
        """
        Takes two tensors of shape (batch, ....)

        Returns one number, so aggregate at end tf.sum or tf.mean(, axis=0)
        """
        
        masks = y_true[:, :, :, self.K:]
        true_values = y_true[:, :, :, :self.K]
        # print("\nMax and min in y_true: ", np.max(true_values), np.min(true_values))
        # print("max in min values in y_pred", np.max(y_pred), np.min(y_pred))

        sq_error = tf.square(true_values - y_pred)
        masked_error = sq_error * masks

        masked_values = tf.reduce_sum(masks, axis=[1,2,3])
        loss = tf.reduce_sum(masked_error, axis=[1,2,3]) / (masked_values + 1e-10)

        return tf.reduce_mean(loss)


class MeanSquaredErrorDobsonMasked(Loss):

    mole = 6.02214076e23
    du = 2.687e16

    def __init__(self, w, b):
        super(MSE, self).__init__(name='du')
        self.w = w
        self.b = b

    def call(self, y_true, y_pred):

        K = y_true.shape[-1]//2
        masks = y_true[:,:,:,K:]
        true_values = y_true[:,:,:,:K]
        true_values = true_values * self.w + self.b
        y_pred = y_pred * self.w + self.b
        
        true_values = self.mole_fraction_to_du_o3(true_values)
        y_pred_dob = self.mole_fraction_to_du_o3(y_pred)


        sq_error = tf.square(true_values - y_pred_dob)
        masked_error = sq_error * masks

        masked_values = tf.reduce_sum(masks, axis=[1,2,3])
        loss = tf.reduce_sum(masked_error, axis=[1,2,3]) / (masked_values + 1e-6)

        return tf.reduce_sum(tf.reduce_mean(loss))

    def mole_fraction_to_du_o3(self, x):
        return x * self.mole / self.du



class MeanSquaredErrorDobson(Loss):

    mole = 6.02214076e23
    du = 2.687e16

    def __init__(self, w, b):
        super(MeanSquaredErrorDobson, self).__init__(name='ppmv') 
        self.w = w
        self.b = b

    def call(self, y_true, y_pred):
        y_true = y_true * self.w + self.b
        y_pred = y_pred * self.w + self.b

        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    def mole_fraction_to_du_o3(self, x):
        return x * self.mole / self.du


class MeanRMSEDU(Loss):

    mole = 6.02214076e23
    du = 2.687e16

    # W and b normalisation factors
    def __init__(self, w, b, K):
        super(MeanRMSEDU, self).__init__(name='ppmv')
        self.w = w
        self.b = b
        self.K = K


    def call(self, y_true, y_pred):
        """
        Takes two tensors of shape (batch, ....)

        Returns one number, so aggregate at end tf.sum or tf.mean(, axis=0)
        """
        masks = y_true[:, :, :, self.K:]
        true_values = y_true[:, :, :, :self.K]

        y_pred = y_pred * self.w + self.b
        true_values = true_values * self.w + self.b

        # true_values = self.mole_fraction_to_du_o3(true_values)
        # y_pred = self.mole_fraction_to_du_o3(y_pred)

        sq_error = tf.square(true_values - y_pred)
        masked_error = sq_error * masks

        masked_values = tf.reduce_sum(masks,axis=[1,2,3])
        loss = tf.reduce_sum(masked_error, axis=[1,2,3]) / (masked_values+1e-6)

        return tf.sqrt(tf.reduce_mean(loss))

    def mole_fraction_to_du_o3(self, x):
        return x * self.mole / self.du


import numpy as np

rmse = MeanRMSEDU(1, 0, 4)




# y_true = np.zeros(shape=(1, 2, 2, 4))
# y_true[0][1][0][0] = 1
# y_true[0][1][1][0] = 1

# y_true[0][1][0][1] = 3
# y_true[0][1][1][1] = 2
# y_true[0][0][0][1] = 4

# y_true[0][1][0][2] = 1
# y_true[0][1][1][2] = 1
# y_true[0][0][0][2] = 1

# y_true[0][1][0][3] = 3
# y_true[0][1][1][3] = 2
# y_true[0][0][0][3] = 4

# print("y_true: \n", y_true)

# y_pred = np.zeros(shape=(1, 2, 2, 2))
# y_pred[0][1][0][0] = 2
# y_pred[0][1][1][0] = 3

# y_pred[0][1][0][1] = 5
# y_pred[0, 1, 1, 1] = 3

# print("y_pred:\n", y_pred)

# print(mse.call(y_true, y_pred))