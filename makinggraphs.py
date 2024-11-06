import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# time taken per epoch from training
interpolated_epoch_times = [
    8803, 8587, 8628, 8557, 8534, 8472, 8560, 8560, 8447,
    8193, 8164, 8362, 8672, 8835, 8798, 8688, 8735, 8691, 
    8745, 8562, 8550, 8456, 8668, 8656, 8570
]

masked_epoch_times = [
    7210, 7301, 7395, 7364, 7359, 7248, 7291, 7336, 7289,
    7289, 7423, 7409, 7347, 7462, 7384, 7425, 7456, 7331,
    7410, 7259, 7285, 7530, 7427, 7427, 7242
]

# no. epoch
epochs = np.array(range(1, len(interpolated_epoch_times) + 1)).reshape(-1, 1)

# linear fit
linear_model = LinearRegression()
linear_model.fit(epochs, interpolated_epoch_times)
linear_pred = linear_model.predict(epochs)

# quad fit
poly_features = PolynomialFeatures(degree=2)
epochs_poly = poly_features.fit_transform(epochs)
quadratic_model = LinearRegression()
quadratic_model.fit(epochs_poly, interpolated_epoch_times)
quadratic_pred = quadratic_model.predict(epochs_poly)

total_time_interp = np.sum(interpolated_epoch_times)
total_time_interp /= 60
total_time_interp /= 60

total_time_masked = np.sum(masked_epoch_times)
total_time_masked /= 60
total_time_masked /= 60


# plotting
plt.figure(figsize=(12, 6))
plt.plot(epochs, interpolated_epoch_times, marker='o', label='Interpolated Epoch Times', color='blue')
plt.plot(epochs, masked_epoch_times, marker='x', label='Masked Epoch Times', color='red')

# plt.plot(epochs, linear_pred, label='Linear Fit', color='red', linestyle='--')
# plt.plot(epochs, quadratic_pred, label='Quadratic Fit', color='green', linestyle='--')
plt.title('Total Seconds Taken per Epoch, Masked vs Interpolated')
plt.xlabel('Epoch')
plt.ylabel('Total Seconds')
plt.xticks(range(1, len(interpolated_epoch_times) + 1))
plt.text(0.08, 0.95, f'Total interpolated training time in hours: {total_time_interp:.2f}\nTotal masked training time in hours: {total_time_masked:.2f}', horizontalalignment='left', verticalalignment='top', 
        transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5) )
plt.legend()
plt.tight_layout()

plt.show()



