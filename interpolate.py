# from load_data import DataLoader
import netCDF4 as nc 
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from scipy.interpolate import interp1d

# Data organised as value[lev][lon][lat]

file_path_laptop = "/Users/max/Desktop/Code/COSC385/Data"


def make_map(vector, lon_range, lat_range, lev_values):
    lon_start = lon_range[0]
    lon_end = lon_range[-1]
    lat_start = lat_range[0]
    lat_end = lat_range[-1]
    
    lon = np.arange(lon_start, lon_end + 5, 5)
    lat = np.arange(lat_start, lat_end + 4, 4)
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    for i in range(vector.shape[2]):
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.OCEAN)

        heatmap = ax.pcolormesh(lon_grid, lat_grid, vector[i, :, :].T, transform=ccrs.PlateCarree(), cmap='viridis')

        cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', pad=0.05)
        cbar.set_label('Concentration')

        ax.set_title(f'Concentration Heatmap at Pressure Level {lev_values[i]} for Ozone 2004 day 241')

        plt.show()


lon_range = [-177.5, 177.5]
lat_range = [-88, 88]

files = os.listdir(file_path_laptop)
paths = []

for path in files:
    if path.startswith("MLS"):
        paths.append(os.path.join(file_path_laptop, path))

print(paths)


file = nc.Dataset(paths[0])
data = file.groups['O3 PressureGrid']['value']
lev_arr = file.groups['O3 PressureGrid']['lev'][:]
# lat = file.groups['O3 PressureGrid']['lat'][:]
# lon = file.groups['O3 PressureGrid']['lon'][:]
# print(file.groups['O3 PressureGrid'])
# print("Lon: ", len(lon))
# print("Lat: ", len(lat))
# print("Lev: ", len(lev_arr))

print(data[0].shape)


data = data[0]

# make_map(data, lon_range, lat_range, lev_arr)

masked_arr = data[7, 0, :]
print(masked_arr)

non_masked = np.where(~masked_arr.mask)[0]
masked = np.where(masked_arr.mask)[0]

interpolated_arr = np.interp(masked, non_masked, masked_arr[~masked_arr.mask])

print(interpolated_arr)

"""
Here we iterate across each pressure level and longitude, 
Then extract each 1d masked array across latitude
Obtain the indicies of masked and non-masked values (~ is not)
If there are values we can use to interpolate we pass it to our 1d interpolation function
then replace the masked values of the array with the interpolated values
"""
for lev in range(data.shape[0]):
    for lon in range(data.shape[1]):
        masked_arr = data[lev, lon, :]  

        if np.ma.is_masked(masked_arr):
            non_masked = np.where(~masked_arr.mask)[0]
            masked = np.where(masked_arr.mask)[0]

            if non_masked.size > 0:
                interpolated_arr = np.interp(masked, non_masked, masked_arr[non_masked])
                # interpolated_arr = interp1d()

                data[lev, lon, masked] = interpolated_arr
            else:
                pass

make_map(data, lon_range, lat_range, lev_arr)

