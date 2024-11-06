import matplotlib.pyplot as plt

train_loss = []
train_ppmv = []
val_loss = []
val_ppmv = []

# Define the output text as a multi-line string
output_text = """
 Epoch 1/25
262/262 [==============================] - 7210s 28s/step - loss: 1.5234e-04 - ppmv: 1.4951e-08 - val_loss: 8.3165e-08 - val_ppmv: 1.7667e-09
Epoch 2/25
262/262 [==============================] - 7301s 28s/step - loss: 5.0510e-11 - ppmv: 3.5725e-11 - val_loss: 1.4598e-11 - val_ppmv: 1.6186e-11
Epoch 3/25
262/262 [==============================] - 7395s 28s/step - loss: 1.3604e-11 - ppmv: 1.4351e-11 - val_loss: 9.4819e-12 - val_ppmv: 8.6156e-12
Epoch 4/25
262/262 [==============================] - 7364s 28s/step - loss: 5.9095e-11 - ppmv: 1.0174e-11 - val_loss: 1.7877e-11 - val_ppmv: 1.3251e-11
Epoch 5/25
262/262 [==============================] - 7359s 28s/step - loss: 4.3574e-13 - ppmv: 2.8545e-12 - val_loss: 1.0384e-11 - val_ppmv: 4.9829e-12
Epoch 6/25
262/262 [==============================] - 7248s 28s/step - loss: 9.2541e-14 - ppmv: 1.0110e-12 - val_loss: 9.6555e-12 - val_ppmv: 4.2916e-12
Epoch 7/25
262/262 [==============================] - 7291s 28s/step - loss: 1.1181e-12 - ppmv: 1.3275e-12 - val_loss: 1.2832e-11 - val_ppmv: 4.6388e-12
Epoch 8/25
262/262 [==============================] - 7336s 28s/step - loss: 3.3690e-10 - ppmv: 1.6345e-11 - val_loss: 2.9499e-09 - val_ppmv: 1.3267e-10
Epoch 9/25
262/262 [==============================] - 7289s 28s/step - loss: 1.1115e-11 - ppmv: 3.1290e-12 - val_loss: 9.4583e-15 - val_ppmv: 1.5020e-13
Epoch 10/25
262/262 [==============================] - 7289s 28s/step - loss: 4.7520e-16 - ppmv: 6.4013e-14 - val_loss: 6.1147e-13 - val_ppmv: 7.2038e-13
Epoch 11/25
262/262 [==============================] - 7423s 28s/step - loss: 7.4980e-13 - ppmv: 6.5812e-13 - val_loss: 5.8208e-14 - val_ppmv: 1.5731e-12
Epoch 12/25
262/262 [==============================] - 7409s 28s/step - loss: 2.5224e-15 - ppmv: 1.7205e-13 - val_loss: 1.8953e-17 - val_ppmv: 1.1924e-14
Epoch 13/25
262/262 [==============================] - 7347s 28s/step - loss: 9.7204e-18 - ppmv: 8.5197e-15 - val_loss: 6.8962e-18 - val_ppmv: 7.5851e-15
Epoch 14/25
262/262 [==============================] - 7463s 29s/step - loss: 3.0224e-18 - ppmv: 4.4772e-15 - val_loss: 2.8175e-18 - val_ppmv: 4.1817e-15
Epoch 15/25
262/262 [==============================] - 7384s 28s/step - loss: 1.2067e-13 - ppmv: 2.4978e-13 - val_loss: 1.5636e-14 - val_ppmv: 1.0305e-12
Epoch 16/25
262/262 [==============================] - 7425s 28s/step - loss: 7.2205e-16 - ppmv: 8.8132e-14 - val_loss: 6.3709e-19 - val_ppmv: 2.0326e-15
Epoch 17/25
262/262 [==============================] - 7456s 28s/step - loss: 1.8152e-19 - ppmv: 8.9217e-16 - val_loss: 2.8090e-19 - val_ppmv: 1.2983e-15
Epoch 18/25
262/262 [==============================] - 7331s 28s/step - loss: 8.5001e-20 - ppmv: 5.5416e-16 - val_loss: 1.4256e-19 - val_ppmv: 8.6109e-16
Epoch 19/25
262/262 [==============================] - 7410s 28s/step - loss: 4.0851e-20 - ppmv: 3.4180e-16 - val_loss: 7.0298e-20 - val_ppmv: 6.0517e-16
Epoch 20/25
262/262 [==============================] - 7259s 28s/step - loss: 1.9921e-20 - ppmv: 2.2790e-16 - val_loss: 3.3164e-20 - val_ppmv: 4.0790e-16
Epoch 21/25
262/262 [==============================] - 7285s 28s/step - loss: 9.7670e-21 - ppmv: 1.4426e-16 - val_loss: 1.6483e-20 - val_ppmv: 2.8242e-16
Epoch 22/25
262/262 [==============================] - 7530s 29s/step - loss: 4.8857e-21 - ppmv: 1.0037e-16 - val_loss: 8.2143e-21 - val_ppmv: 1.8602e-16
Epoch 23/25
262/262 [==============================] - 7427s 28s/step - loss: 2.3889e-21 - ppmv: 7.2520e-17 - val_loss: 4.0685e-21 - val_ppmv: 1.3460e-16
Epoch 24/25
262/262 [==============================] - 7427s 28s/step - loss: 1.2034e-21 - ppmv: 5.1901e-17 - val_loss: 2.0166e-21 - val_ppmv: 1.0275e-16
Epoch 25/25
262/262 [==============================] - 7242s 28s/step - loss: 5.9766e-22 - ppmv: 3.7446e-17 - val_loss: 9.9366e-22 - val_ppmv: 8.0320e-17
"""

# Parse each line to extract values
for line in output_text.strip().splitlines():
    if "Epoch" not in line:
        # Split the line based on spaces and '=' to access values
        parts = line.split()
        train_loss.append(float(parts[7]))
        train_ppmv.append(float(parts[10]))
        val_loss.append(float(parts[13]))
        val_ppmv.append(float(parts[-1]))

# The resulting arrays
print("Train Loss:", train_loss)
print("Train PPMV:", train_ppmv)
print("Validation Loss:", val_loss)
print("Validation PPMV:", val_ppmv)

history = {}

history['loss'] = train_loss
history['ppmv'] = train_ppmv
history['val_loss'] = val_loss
history['val_ppmv'] = val_ppmv

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Training and validation loss masked model')
plt.ylabel('log MSE loss')
plt.xlabel('Epoch')
plt.yscale('log')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.plot(history['ppmv'])
plt.plot(history['val_ppmv'])
plt.title('Normalised RMSE Loss masked model')
plt.ylabel('log loss (ppmv)')
plt.xlabel('epoch')
plt.yscale('log')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()