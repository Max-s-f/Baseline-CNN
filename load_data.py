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
from datetime import datetime, timedelta


class DataGenerator(keras.utils.Sequence):

    mole = 6.02214076e23
    du = 2.687e16


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
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.lev_range = lev_range
        self.cache_dir = cache_dir
        self.lat_range.sort()
        self.lon_range.sort()
        self.lev_range.sort()
        self.num_samples = len(self.X_files)
        self.on_epoch_end()

        if norm_dict is not None:
            self.norm_dict = norm_dict
        elif normalise:
            self.__set_norm()


    def on_epoch_end(self):
        """
        Updates indexes after each epoch and shuffles if specified.
        """
        self.indexes = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # def round_lat(self):
    #     lat_keys = self.lat_to_index.keys()
    #     min_range = min(lat_keys, key=lambda x: abs(x - self.lat_range[0]))
    #     self.lat_range[0] = min_range

    #     max_range = min
    
    
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
        Returns:
        - norm_dict: Dictionary with normalization parameters.
        """
        return self.norm_dict


    def __data_generation(self, list_IDs_temp):
        """
        Generates data for the current batch.

        Parameters:
        - list_IDs_temp: List of sample indexes for the current batch.

        Returns:
        - batch_X: Input data for the batch.
        - batch_y: Target data for the batch.
        """
        batch_X = []
        batch_y = []

        for ID in list_IDs_temp:
            X_data = []
            X_masks = []
            y_data = []
            y_masks = []

            for i, tuple in enumerate(self.X_files[ID]):
                for i, file in enumerate(tuple):
                    file_data, masks = self.__get_file(file, self.measurements[i])

                    for data in file_data:
                        X_data.append(data)

                    for mask in masks:
                        X_masks.append(mask)

            for i, file in enumerate(self.y_files[ID]):
                file_data, masks = self.__get_file(file, self.target[0])

                for data in file_data:
                    y_data.append(data)

                for mask in masks:
                    y_masks.append(mask)

            X_data = np.array(X_data)
            y_data = np.array(y_data)

            # Set any value that equals -999.99 to 0
            X_data = np.where(X_data == -999.99, 0, X_data)
            y_data = np.where(y_data == -999.99, 0, y_data)

            X_data_stacked = np.stack(X_data, axis=0)
            X_masks_stacked = np.stack(X_masks, axis=0)
            X = np.concatenate([X_data_stacked, X_masks_stacked], axis=0)

            y_data_stacked = np.stack(y_data, axis=0)
            y_masks_stacked = np.stack(y_masks, axis=0)
            y = np.concatenate([y_data_stacked, y_masks_stacked], axis=0)
            
            batch_X.append(X)
            batch_y.append(y)
        return batch_X, batch_y


    def __get_file(self, file_path, measure):
        """
        Retrieves and processes data from a specified file.

        Parameters:
        - file_path: Path to the file.
        - measure: Measurement to retrieve from the file.

        Returns:
        - final_data: Processed data from the file.
        """
        cached_files = os.listdir(self.cache_dir)
        # boolean to set to true if we retrieve data from cache, otherwise we cache the file_data and close it 
        pickled = False
        pickle_file = None
        file_name = file_path.split('\\')[-1]
        if file_path.endswith('.nc'):
            pickle_file = file_name[:-3]
            pickle_file += '.pkl'
        pickle_path = os.path.join(self.cache_dir, pickle_file)


        if pickle_file in cached_files:
            cached_file = None
            with open(pickle_path, 'rb') as f:
                cached_file = pickle.load(f)
            lev = cached_file[0]
            data = cached_file[1]
            pickled = True

        if not pickled:
            file = nc.Dataset(file_path)
            data = file.groups[f'{measure} PressureGrid']['value'][:]
            lev = file.groups[f'{measure} PressureGrid']['lev'][:]

            cached_file = [lev, data]
            with open(pickle_path, 'wb') as f:
                pickle.dump(cached_file, f)
            file.close()

        final_data = []
        masks = []

        lat_start_idx = self.lat_to_index[self.lat_range[0]]
        lat_end_idx = self.lat_to_index[self.lat_range[1]] + 1
        lon_start_idx = self.lon_to_index[self.lon_range[0]]
        lon_end_idx = self.lon_to_index[self.lon_range[1]] + 1

        sliced_data = data[:, :, lon_start_idx:lon_end_idx, lat_start_idx:lat_end_idx]
        # sliced_data = np.ma.masked_less_equal(sliced_data, 0)
        normalised_sliced_data = sliced_data

        if self.normalise:
            normalised_sliced_data -= self.norm_dict[measure][0]
            normalised_sliced_data /= self.norm_dict[measure][1]

        for i in range(len(normalised_sliced_data[0])):
            slice_data = normalised_sliced_data[0][i]

            if self.lev_range[0] <= lev[i] <= self.lev_range[1]:
                mask = slice_data.mask
                masks.append(mask)
                
                final_data.append(slice_data)
        
        return final_data, masks


    def __getitem__(self, index):
        """
        Generates one batch of data.

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

    def make_map(self, vector, measurement):
        """
        Generates a heatmap for a specified measurement.

        Parameters:
        - vector: Data vector for the measurement.
        - measurement: The measurement to plot.

        Generates and shows a heatmap.
        """
        lon_start = self.lon_range[0]
        lon_end = self.lon_range[-1]
        lat_start = self.lat_range[0]
        lat_end = self.lat_range[-1]
        
        lon = np.arange(lon_start, lon_end + 5, 5)
        lat = np.arange(lat_start, lat_end + 4, 4)
        
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        for i in range(vector.shape[2]):
            fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.LAND, edgecolor='black')
            ax.add_feature(cfeature.OCEAN)

            # Use lon_grid and lat_grid directly in pcolormesh
            heatmap = ax.pcolormesh(lon_grid, lat_grid, vector[:, :, i].T, transform=ccrs.PlateCarree(), cmap='viridis')

            cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', pad=0.05)
            cbar.set_label(f'{measurement} Concentration')

            ax.set_title(f'{measurement} Concentration Heatmap')

            plt.show()

class DataLoader():

    file_dict = {}
    target_dict = {}


    def __init__(self, time_period = [2004248, 20024010], target = ['O3'], time_step = 1,
                measurements = ['SO2', 'CO', 'H2O', 'HCl', 'HNO3', 'N2O', 'Temperature'], 
                directory_path=r'\Users\simma362\Desktop\Code\Baseline-CNN\Data'):
        self.directory_path = directory_path
        self.time_step = time_step
        self.measurements = measurements
        self.target = target
        self.paths = self.__make_paths(measurements)
        self.target_path = self.__make_paths(target)
        self.time_period = time_period


    def __make_paths(self, measurements):
        paths = []
        for m in measurements:
            path = os.path.join(self.directory_path, m)
            paths.append(path)
        return paths


    """
    We need to sort the files based on only the year and date
    Otherwise the other characters in the filename mess it up
    """
    def __get_sort_key(self, s):
            parts = s.split('_')
            date_part = parts[-1]
            year = date_part[:4]
            day = date_part[5:8]
            key = year+day
            return key

    """
    function  to convert year_day in file to datetime
    makes it easy to handle finding file that is time_step days forwards

    params:
    - file - file to return date time object of 
    """
    def __convert_to_date(self, file):
        parts = file.split('_')[-1]
        year = int(parts[:4])
        day = int(parts[5:8])
        return datetime(year, 1, 1) + timedelta(days=day - 1)



    """
    Method to create a dictionary for each measurement according to dates specified by user
    dict[measurement] = list(files containing sorted data between specified dates)
    """
    def get_files(self):
        start_date = str(self.time_period[0])
        start_date = start_date[:4] + 'd' + start_date[4:]
        end_date = str(self.time_period[1])
        end_date = end_date[:4] + 'd' + end_date[4:]

        for i, path in enumerate(self.paths):
            files = os.listdir(path)
            files.sort(key=self.__get_sort_key)

            start_index = next((i for i, s in enumerate(files) if start_date in s), None)
            end_index = next((i for i, s in enumerate(files) if end_date in s), None)

            if start_index is not None and end_index is not None:
                files = files[start_index:end_index+1]

            for x in range(len(files)):
                files[x] = os.path.join(path, files[x])

            self.file_dict[self.measurements[i]] = files
        
        for i, path in enumerate(self.target_path):
            files = os.listdir(path)
            files.sort(key=self.__get_sort_key)

            start_index = next((i for i, s in enumerate(files) if start_date in s), None)
            end_index = next((i for i, s in enumerate(files) if end_date in s), None)

            if start_index is not None and end_index is not None:
                files = files[start_index:end_index+1]

            for x in range(len(files)):
                files[x] = os.path.join(path, files[x])

            self.target_dict[self.target[i]] = files

        X_files = list(zip(*self.file_dict.values()))
        y_files = list(zip(*self.target_dict.values()))

        final_X_files = []
        final_y_files = []
        for file_set in X_files:
            date = self.__convert_to_date(file_set[0])
            target_date = date + timedelta(days = self.time_step)
            # Try find matching file time_step days ahead
            next_X_file = None
            next_y_file = None
            for file_pair in  zip(X_files, y_files):
                X_file = file_pair[0]
                y_file = file_pair[1]
                next_x_date = self.__convert_to_date(X_file[0])
                next_y_date = self.__convert_to_date(y_file[0])

                # If we find file time_step days ahead we set it to next file and break
                if next_x_date == target_date:
                    next_X_file = X_file
                
                if next_y_date == target_date:
                    next_y_file = y_file
                
                if next_y_file and next_X_file:
                    break
            # We have found X_file, now just add y_file time_step days ahead
            if next_X_file and next_y_file:
                final_X_files.append((file_set, next_X_file))
                final_y_files.append(next_y_file)


        return final_X_files, final_y_files


if __name__ == "__main__":
    # data_loader = DataLoader()
    # x_files, y_files = data_loader.get_files()
    # training_data = DataGenerator(x_files, y_files, lat_range=[-88, 88], lon_range=[-177.5, 177.5])
    # batch_X, batch_y = training_data[0]
    # # print(len(training_data))
    # print(batch_X.shape)
    measurements = ['SO2', 'CO', 'H2O', 'HCl', 'HNO3', 'N2O', 'Temperature']
    time = [2021010, 2024010]
    lat = [0, 84]
    lon = [2.5, 172.5]

    data_loader = DataLoader(measurements = measurements, time_step=30)
    X_files, y_files = data_loader.get_files()

    training_data = DataGenerator(X_files, y_files, normalise_sample = 100)
    # print(X_files[0], y_files[0])

    for batch_X, batch_y in training_data:
        print(batch_X.shape, batch_y.shape)
        print(np.min(batch_y), np.max(batch_y))
        print(np.min(batch_X), np.max(batch_X))



    # for i, fileset in enumerate(X_files):
    #     first_file = fileset[0]
    #     last_file = fileset[-1]
    #     parts = first_file[0].split('_')
    #     date_part = parts[-1]
    #     year = date_part[:4]
    #     day = date_part[5:8]
    #     first_key = year+day

    #     parts = last_file[0].split('_')
    #     date_part = parts[-1]
    #     year = date_part[:4]
    #     day = date_part[5:8]
    #     last_key = year+day

    #     if int(first_key) != int(last_key)-30:
    #         print(f"ERROR for index {i}\nFirst key: {first_key}\nLast key: {last_key}")


    # print(len(y_files), len(X_files))

    # training_data = DataGenerator(X_files, y_files, normalise=True, normalise_sample=500)
    # batch_X, batch_y = training_data[0]

    # print(batch_X.shape)
    # print(np.max(batch_X), np.max(batch_y))
    # print(np.min(batch_X), np.min(batch_y))

    # for batch_X, batch_y in training_data:
        
        # training_data.make_map(batch_X[0], 'Measure')
