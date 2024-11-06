import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets, layers, models
from load_data import DataLoader, DataGenerator
from CustomMSE import MeanSquaredErrorX
from CustomCall import LoggingCallback

measurements = ['SO2', 'CO', 'H2O', 'HCl', 'HNO3', 'N2O', 'Temperature']
time = [2005001, 2020150]
lon = [-177.5, 177.5]
lat = [-88, 88]

dataloader = DataLoader(measurements=measurements, time_period=time)
X_files, y_files = dataloader.get_files()


train_size = len(X_files) * 6 // 10
validation_size = len(X_files) * 2 // 10
test_size = len(X_files) * 2 // 10


X_train = X_files[:train_size]
y_train = y_files[:train_size]

X_validation = X_files[train_size:train_size + validation_size]
y_validation = y_files[train_size:train_size + validation_size]

X_test = X_files[train_size + validation_size:]
y_test = y_files[train_size + validation_size:]

if not os.path.isdir("saved"):
    os.mkdir("saved")

# Save names of files
save_name = os.path.join('saved', 'first_cnn_model')
checkpoint_save_name = save_name + '_cnn_net.chk'

training_data = DataGenerator(X_files=X_train, y_files=y_train, batch_size=128, shuffle=True, data_aug=False, normalise=True, normalise_sample=5000)

norm_dict = training_data.get_norm_dict()

validation_data = DataGenerator(X_files=X_validation, y_files=y_validation, batch_size=128, shuffle=True, data_aug=False, normalise=True, norm_dict=norm_dict)
test_data = DataGenerator(X_test, y_test, batch_size=128, shuffle=False, data_aug=False, normalise=False)

# for i in range(len(training_data)):
#     batch_X, batch_y = training_data[i]
#     print('Batch X shape:', batch_X.shape)
#     print('Batch y shape:', batch_y.shape)

batch_X, batch_y = training_data[0]
input_shape = batch_X.shape[1:]
output_shape = batch_y.shape[1:]
H, W, C = output_shape[0], output_shape[1], output_shape[2]
C = C // 2
output_shape = (H, W, C)

model = models.Sequential()
model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='same', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(H * W * C, activation='linear'))  

model.add(layers.Reshape(output_shape))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=MeanSquaredErrorX())

model.summary()

loggingCallback = LoggingCallback(model=model)
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath='current_best',
#     save_weights_only=True,
#     monitor='val_loss',
#     mode='min',
#     save_best_only=True)

model.fit(training_data,
        validation_data=validation_data,
        epochs=20)

metrics = model.evaluate(test_data)
print(metrics)

# Predict using the test data
batch_X, batch_y = test_data[0]
predictions = model.predict(batch_X)
print(f'Test Predictions: {predictions}')

if len(predictions.shape) == 4:
    print('called')
    training_data.make_map(predictions[0], 'O3')
else:
    print('made it')
    training_data.make_map(predictions, 'O3')