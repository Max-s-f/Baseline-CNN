import netCDF4 as nc
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import progressbar
import sys
import pickle
from load_data import DataLoader

class InterpolatedDataGenerator(keras.utils.Sequence):

    lat_to_index = {-88.0: 0, -84.0: 1, -80.0: 2, -76.0: 3, -72.0: 4, -68.0: 5, -64.0: 6, -60.0: 7, -56.0: 8, -52.0: 9, -48.0: 10, -44.0: 11, -40.0: 12, 
                -36.0: 13, -32.0: 14, -28.0: 15, -24.0: 16, -20.0: 17, -16.0: 18, -12.0: 19, -8.0: 20, -4.0: 21, 0.0: 22, 4.0: 23, 8.0: 24, 12.0: 25, 
                16.0: 26, 20.0: 27, 24.0: 28, 28.0: 29, 32.0: 30, 36.0: 31, 40.0: 32, 44.0: 33, 48.0: 34, 52.0: 35, 56.0: 36, 60.0: 37, 64.0: 38, 68.0: 
                39, 72.0: 40, 76.0: 41, 80.0: 42, 84.0: 43, 88.0: 44}

    lon_to_index = {-177.5: 0, -172.5: 1, -167.5: 2, -162.5: 3, -157.5: 4, -152.5: 5, -147.5: 6, -142.5: 7, -137.5: 8, -132.5: 9, -127.5: 10, -122.5: 11, 
                    -117.5: 12, -112.5: 13, -107.5: 14, -102.5: 15, -97.5: 16, -92.5: 17, -87.5: 18, -82.5: 19, -77.5: 20, -72.5: 21, -67.5: 22, -62.5: 23, 
                    -57.5: 24, -52.5: 25, -47.5: 26, -42.5: 27, -37.5: 28, -32.5: 29, -27.5: 30, -22.5: 31, -17.5: 32, -12.5: 33, -7.5: 34, -2.5: 35, 2.5: 36, 
                    7.5: 37, 12.5: 38, 17.5: 39, 22.5: 40, 27.5: 41, 32.5: 42, 37.5: 43, 42.5: 44, 47.5: 45, 52.5: 46, 57.5: 47, 62.5: 48, 67.5: 49, 72.5: 50, 
                    77.5: 51, 82.5: 52, 87.5: 53, 92.5: 54, 97.5: 55, 102.5: 56, 107.5: 57, 112.5: 58, 117.5: 59, 122.5: 60, 127.5: 61, 132.5: 62, 137.5: 63, 
                    142.5: 64, 147.5: 65, 152.5: 66, 157.5: 67, 162.5: 68, 167.5: 69, 172.5: 70, 177.5: 71}

    norm_dict = {}


    def __init__(self, X_files, y_files, batch_size = 128, shuffle = True, data_aug = True, normalise = True, normalise_sample = 500, norm_dict = None, cache_dir = 'picklecache/', lat_range = [-88, 88], lon_range = [-177.5, 177.5], lev_range = [0.0004, 216.0], 
                target = ['O3'], measurements = ['SO2', 'CO', 'H2O', 'HCl', 'HNO3', 'N2O', 'Temperature']):
        """
        Initialization of the DataGenerator class.

        Parameters:
        - X_files: List of file paths for the input data.
        - y_files: List of file paths for the target data.
        - batch_size: Number of samples per batch.
        - shuffle: Whether to shuffle the data at the end of each epoch.
        - data_aug: Whether to apply data augmentation.
        - normalise: Whether to normalize the data.
        - normalise_sample: Number of samples to use for normalization.
        - norm_dict: Dictionary for normalization parameters.
        - lat_range: Latitude range to consider.
        - lon_range: Longitude range to consider.
        - lev_range: Level range to consider.
        - target: List of target measurements.
        - measurements: List of input measurements.
        """
        self.X_files = X_files
        self.y_files = y_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_aug = data_aug
        self.normalise = normalise
        self.normalise_sample = normalise_sample
        self.measurements = measurements
        self.target = target
        self.cache_dir = cache_dir
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.lev_range = lev_range
        self.lat_range.sort()
        self.lon_range.sort()
        self.lev_range.sort()
        self.num_samples = len(self.X_files)
        self.on_epoch_end()
        # self.expected_X_channels, self.expected_y_channels = self.__estimate_channels()
        if norm_dict is not None:
            self.norm_dict = norm_dict
        elif normalise:
            self.__set_norm()

    
    def on_epoch_end(self):
        self.indexes = np.arange(1, self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)
    

    # def __estimate_channels(self):
    #     expected_X_channels = 0
    #     expected_y_channels = 0

    #     for file in X_files[len(X_files)//2][0]:
    #         with nc.Dataset(file) as file_data:
    #             data = file_data.groups[f'{m} PressureGrid']['value'][:]
    #             data = data[0]

    #             lat_start_idx = self.lat_to_index[self.lat_range[0]]
    #             lat_end_idx = self.lat_to_index[self.lat_range[1]] + 1
    #             lon_start_idx = self.lon_to_index[self.lon_range[0]]
    #             lon_end_idx = self.lon_to_index[self.lon_range[1]] + 1
    #             sliced_data = data[:, lon_start_idx:lon_end_idx, lat_start_idx:lat_end_idx]

    #             lev_arr = file_data.groups[f'{m} PressureGrid']['lev'][:]
    #             for lev in range(sliced_data.shape[0]):
    #                 if self.lev_range[0] <= lev_arr[lev] <= self.lev_range[1]:
    #                                 expected_X_channels += 1
        
    #     for file in y_files[len(y_files)//2]:
    #         print(file)
    #         with nc.Dataset(file) as file_data:
    #             for m in self.target:
    #                 group_keys = file_data.groups.keys()
    #                 if f"{m} PressureGrid" in group_keys:
    #                     lev_arr = file_data.groups[f'{m} PressureGrid']['lev'][:]
    #                     for lev in lev_arr:
    #                         if self.lev_range[0] <= lev <= self.lev_range[1]:
    #                             expected_y_channels += 1
    
    #     print(f'expected x channels: {expected_X_channels}\nexpected y channels {expected_y_channels}')
    #     return expected_X_channels, expected_y_channels


    def __set_norm(self):
        """
        Sets normalization parameters for each measurement and target.
        """
        # Edge case where set no. samples per measure is greater than the no. files provided per measure
        if len(self.X_files) * (len(self.measurements) + 1) < self.normalise_sample:
            self.normalise_sample = len(self.X_files) // len(self.measurements)

        for m in self.measurements:
            self.norm_dict[m] = [0, 0]
        
        for t in self.target:
            self.norm_dict[t] = [0, 0]

        all_files = [a + b[0] + b[1] for a, b in zip(self.y_files, self.X_files)]
        all_measures = self.measurements + self.target
        np.random.shuffle(all_files)

        all_slices_dict = {m: [] for m in all_measures}

        for i in progressbar.progressbar(range(self.normalise_sample)):
            fileset = all_files[i]
            for data in fileset:
                try:
                    file = nc.Dataset(data)
                except:
                    sys.stderr.write("Failed on {}\n".format(data))
                    continue
    
                for m in all_measures:
                    group_keys = file.groups.keys()
                    if f"{m} PressureGrid" in group_keys:
                        data = file.groups[f'{m} PressureGrid']['value'][:]


                        data = np.ma.masked_less(data, 0)

                        lat_start_idx = self.lat_to_index[self.lat_range[0]]
                        lat_end_idx = self.lat_to_index[self.lat_range[1]] + 1
                        lon_start_idx = self.lon_to_index[self.lon_range[0]]
                        lon_end_idx = self.lon_to_index[self.lon_range[1]] + 1

                        sliced_data = data[:, :, lon_start_idx:lon_end_idx, lat_start_idx:lat_end_idx]
                        all_slices_dict[m].append(sliced_data)

                file.close()

        for m in all_measures:
            all_slices = np.ma.concatenate(all_slices_dict[m], axis=0)
            self.norm_dict[m][0] = np.ma.mean(all_slices)
            self.norm_dict[m][1] = np.ma.max(np.ma.abs(all_slices - self.norm_dict[m][0]))
        

    def get_norm_dict(self):
        """
        Method to return norm_dict so we can have same normalisation across 
        testing, validation and training datasets
        @return norm_dict
        """
        return self.norm_dict


    def __data_generation(self, list_IDs_temp):
        batch_X = []
        batch_y = []

        for ID in list_IDs_temp:
            X = []
            y = []

            for i, tuple in enumerate(self.X_files[ID]):
                for i, file in enumerate(tuple):
                    file_data = self.__get_file__(file, self.measurements[i])

                    for data in file_data:
                        X.append(data)

            
            for i, file in enumerate(self.y_files[ID]):
                file_data = self.__get_file__(file, self.target[0])

                for data in file_data:
                    y.append(data)

            X = np.array(X)
            y = np.array(y)

            X = np.where(X == -999.99, 0, X)
            y = np.where(y == -999.99, 0, y)

            X = np.stack(X, axis=0)
            y = np.stack(y, axis=0)


            channels = X.shape[0]
            if channels != 322:
                # print(f"Incorrect no. channels for index {index}")
                missing_channels = 322 - channels
                X = np.concatenate([X, X[-missing_channels:, :, :]], axis=0)


            batch_X.append(X)
            batch_y.append(y)
        return batch_X, batch_y


    def __get_file__(self, file_path, measure):
        cached_files = os.listdir(self.cache_dir)
        # boolean to set to true if we retrieve data from cache, otherwise we cache the file_data and close it 
        pickled = False
        pickle_file = None
        split_path = file_path.split('\\')
        file_name = split_path[-1]
        if file_path.endswith('.nc'):
            pickle_file = file_name[:-3]
            pickle_file += '.pkl'
        
        pickle_path = os.path.join(self.cache_dir, pickle_file)


        if pickle_file in cached_files:
            cached_file = None
            with open(pickle_path, 'rb') as f:
                cached_file = pickle.load(f)
            lev_arr = cached_file[0]
            data = cached_file[1]
            pickled = True

        if not pickled:
            file = nc.Dataset(file_path)
            data = file.groups[f'{measure} PressureGrid']['value'][:]
            lev_arr = file.groups[f'{measure} PressureGrid']['lev'][:]

            cached_file = [lev_arr, data]
            with open(pickle_path, 'wb') as f:
                pickle.dump(cached_file, f)
            file.close()

        final_data = []
        data = data[0]

        lat_start_idx = self.lat_to_index[self.lat_range[0]]
        lat_end_idx = self.lat_to_index[self.lat_range[1]] + 1
        lon_start_idx = self.lon_to_index[self.lon_range[0]]
        lon_end_idx = self.lon_to_index[self.lon_range[1]] + 1

        sliced_data = data[:, lon_start_idx:lon_end_idx, lat_start_idx:lat_end_idx]

        sliced_data = np.ma.masked_less_equal(sliced_data, 0)

        for lev in range(sliced_data.shape[0]):
            add_slice = False

            if self.lev_range[0] <= lev_arr[lev] <= self.lev_range[1]:
                for lat in range(sliced_data.shape[2]):
                    masked_arr = sliced_data[lev, :, lat]

                    if np.ma.is_masked(masked_arr):
                        non_masked = np.where(~masked_arr.mask)[0]
                        masked = np.where(masked_arr.mask)[0]

                        if non_masked.size > 0:
                            interpolated_arr = np.copy(masked_arr)
                            interpolated_values = np.interp(masked, non_masked, masked_arr[non_masked])
                            add_slice = True
                            interpolated_arr[masked] = interpolated_values

                            sliced_data[lev, :, lat] = interpolated_arr
                        else:
                            # This error message will pop up heaps so commented to avoid spam 
                            # sys.stderr.write("Array with no non masked values found")
                            continue

                if self.normalise:
                    sliced_data[lev] -= self.norm_dict[measure][0]
                    sliced_data[lev] /= self.norm_dict[measure][1]

                if self.normalise and add_slice and np.min(sliced_data) >= -100:
                    final_data.append(sliced_data[lev])
                    add_slice = False

        final_data = np.array(final_data)

            # if final_data.shape[-1] != 45:
            #     print("BREAK HERE", final_data.shape, file_path)
        return final_data


    def __getitem__(self, index):
        """
        Generates one batch of data.
        Currently hard coded to avoid a channel size bug where channels 
        for one batch would be 307 rather than 308 using all measures
        
        Parameters:
        - index: Index of the batch.

        Returns:
        - X: Input data for the batch.
        - y: Target data for the batch.
        """
        list_IDs_temp = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__data_generation(list_IDs_temp)
        X = np.transpose(X, [0, 2, 3, 1])
        y = np.transpose(y, [0, 2, 3, 1])

        return X, y

    def __len__(self):
        """
        Denotes the number of batches per epoch.

        Returns:
        - Number of batches per epoch.
        """
        return int(np.ceil(self.num_samples / self.batch_size))


    def make_map(self, vector, lon_range, lat_range, titleStr = "O3 Concentration Heatmap"):
        lon_start = lon_range[0]
        lon_end = lon_range[-1]
        lat_start = lat_range[0]
        lat_end = lat_range[-1]

        lon = np.arange(lon_start-5, lon_end, 5)
        lat = np.arange(lat_start-4, lat_end, 4)
        
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        for i in range(vector.shape[-1]):
            fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.LAND, edgecolor='black')
            ax.add_feature(cfeature.OCEAN)

            heatmap = ax.pcolormesh(lon_grid, lat_grid, vector[0, :, :, i].T, transform=ccrs.PlateCarree(), cmap='viridis')

            cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', pad=0.05)
            cbar.set_label('Concentration')

            ax.set_title(titleStr)

            plt.show()


if __name__ == "__main__":
    measurements = ['SO2', 'CO', 'H2O', 'HCl', 'HNO3', 'N2O', 'Temperature']
    time = [2004245, 2024010]
    lon = [-172.5, 172.5]
    lat = [-84, 84]
    lev = [0, 152]

    print(os.listdir('picklecache/'))
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

    training_data = InterpolatedDataGenerator(X_files=X_train, y_files=y_train, batch_size=128, shuffle=True, data_aug=False, normalise=True, normalise_sample=100, lev_range=lev, lon_range=lon, lat_range=lat)

    norm_dict = training_data.get_norm_dict()

    validation_data = InterpolatedDataGenerator(X_files=X_validation, y_files=y_validation, batch_size=128, shuffle=True, data_aug=False, normalise=True, norm_dict=norm_dict, lev_range = lev,  lon_range=lon, lat_range=lat)
    test_data = InterpolatedDataGenerator(X_test, y_test, batch_size=128, shuffle=False, data_aug=False, normalise=True, norm_dict=norm_dict, lev_range=lev,  lon_range=lon, lat_range=lat)



    batch_X, batch_y = training_data[0]
    # print(batch_X.shape)

    for batch_X, batch_y in training_data:
        print(batch_X.shape, batch_y.shape)

    # for batch_X, batch_y in training_data:
    #         value = -999.99
    #         first_lat_X = batch_X[:, :, 0, :]
    #         last_lat_X = batch_X[:, :, -1, :]
            
    #         # Check first and last latitude index (index 0 and -1) for batch_y
    #         first_lat_Y = batch_y[:, :, 0, :]
    #         last_lat_Y = batch_y[:, :, -1, :]
    #         print(first_lat_Y.shape)
            
    #         # Check if all values in the first and last latitude indices are equal to `-999.99` for both batch_X and batch_y
    #         all_first_lat_X = np.all(first_lat_X == value)
    #         all_last_lat_X = np.all(last_lat_X == value)
            
    #         all_first_lat_Y = np.all(first_lat_Y == value)
    #         all_last_lat_Y = np.all(last_lat_Y == value)
    #         print(all_first_lat_X)
    #         print(all_last_lat_X)
    #         print(all_first_lat_Y)
    #         print(all_last_lat_Y)

    # training_data.make_map(batch_y, lon, lat)