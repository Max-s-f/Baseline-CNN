import netCDF4 as nc
import numpy as np
import pandas
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

directory_path = r'\Users\simma362\Desktop\Code\Baseline-CNN\Data\O3'
files = os.listdir(directory_path)


for file in [files[-1]]:
    file_path = os.path.join(directory_path, file)
    data = nc.Dataset(file_path)

# print(data)s
# print(data.variables.keys())
# print(data.variables['OriginalInputFiles'][:])

print(data.groups.keys())

o3_grid = 'O3 PressureGrid'
o3_group = data.groups[o3_grid]

print(o3_group.variables.keys())

ozone = o3_group.variables['value']
lat = o3_group.variables['lat']
lon = o3_group.variables['lon']
lev = o3_group.variables['lev']

lat = lat[:]
print(len(lat))

lon = lon[:]
print(len(lon))

lev = lev[:]
print(len(lev))

ozone_data = ozone[:]
print(ozone_data[0][40][70])
print(ozone_data.shape)

# Ozone data shape: value, lev, lon, lat
ozone_data_2d = ozone_data[0, 30, :, :]

# plt.figure(figsize=(12, 6))
# sns.heatmap(ozone_data_2d, cmap='viridis')
# plt.title('Ozone Concentration Heatmap')
# plt.xlabel('Longitude Index')
# plt.ylabel('Pressure Level Index')
# plt.show()

# # Define the latitude and longitude values
# latitudes = np.linspace(-90, 90, ozone_data_2d.shape[1])
# longitudes = np.linspace(-180, 180, ozone_data_2d.shape[0])

# Create a figure with a cartopy projection
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# Add features to the map
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)

# Create a heatmap
heatmap = ax.pcolormesh(lon, lat, ozone_data_2d.T, transform=ccrs.PlateCarree(), cmap='viridis')

# Add a colorbar
cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', pad=0.05)
cbar.set_label('Ozone Concentration')

# Set the title
ax.set_title(f'Ozone Concentration Heatmap (Pressure Level {lev[30]})')

# Show the plot
plt.show()