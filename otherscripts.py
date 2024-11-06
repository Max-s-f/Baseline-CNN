import os
from load_data import DataLoader, DataGenerator

class FileComparator:
    def __init__(self, file_dict):
        self.file_dict = file_dict
        self.compare_dates()

    def compare_dates(self):
        ref_files_dict = {key: value for key, value in self.file_dict.items()}
        date_year_dict = {}
        for key, files in ref_files_dict.items():
            date_year_set = set()
            for file in files:
                parts = file.split('_')
                date_part = parts[-1]
                year = date_part[:4]
                day = date_part[5:8]
                date_year_set.add(year + day)
            date_year_dict[key] = date_year_set
        
        all_keys = list(ref_files_dict.keys())
        for i in range(len(all_keys)):
            for j in range(i + 1, len(all_keys)):
                key_i = all_keys[i]
                key_j = all_keys[j]
                missing_dates_i = date_year_dict[key_j] - date_year_dict[key_i]
                missing_dates_j = date_year_dict[key_i] - date_year_dict[key_j]

                if missing_dates_i:
                    print(f'Missing in {key_i} for dates present in {key_j}: {missing_dates_i}')
                if missing_dates_j:
                    print(f'Missing in {key_j} for dates present in {key_i}: {missing_dates_j}')



measurements = measurements = ['SO2', 'CO', 'H2O', 'HCl', 'HNO3', 'N2O', 'Temperature']


dataloader = DataLoader(measurements=measurements)
X_files, y_files = dataloader.get_files()

file_dict = dataloader.file_dict
target_dict = dataloader.target_dict

file_dict['O3'] = target_dict['O3']

# filecompare = FileComparator(file_dict)


# print(len(X_files), len(y_files))

all_data = DataGenerator(X_files, y_files, normalise=False)

batch_X, batch_y = all_data[0]

print(len(batch_X))
print(batch_y.shape)
all_data.make_map(batch_y[0], 'O3')

"""
This is a script to check what files are missing:
        
        files = self.file_dict['SO2']
        for key, value in self.file_dict.items():
            if key == 'SO2':
                continue
            for i, s in enumerate(value):
                parts = s.split('_')
                other_parts = files[i].split('_')
                other_date = other_parts[-1]
                other_year = other_date[:4]
                other_day = other_date[5:8]
                other_key = other_year + other_day
                date_part = parts[-1]
                year = date_part[:4]
                day = date_part[5:8]
                key = year+day
                if other_key != key:
                    print(key, other_key)
                    

                    insert at the end of __file_dict in load_data()

The 33rd day of 2024 is February 2, 2024.
The 116th day of 2010 is April 26, 2010.
The 217th day of 2020 is August 4, 2020.                    
"""