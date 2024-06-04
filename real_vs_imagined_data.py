import numpy as np
import os

# Define the directory containing the .npy files
data_dir = 'data/'

# Define the real and imagined file numbers
real_files = ['S001_S001R03.edf.npy', 'S001_S001R07.edf.npy'
             , 'S001_S001R11.edf.npy']
imagined_files = ['S001_S001R04.edf.npy', 'S001_S001R08.edf.npy',
                 'S001_S001R12.edf.npy']

# Helper function to load and concatenate .npy files
def load_and_concatenate(file_list, directory):
    data_list = []
    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            data = np.load(file_path)
            data_list.append(data)
        else:
            print(f"File {file_path} does not exist")
    return np.concatenate(data_list, axis=0)  # Adjust axis if necessary

# Combine real data files
real_data = load_and_concatenate(real_files, data_dir)
np.save(os.path.join(data_dir, 'real_data_fists.npy'), real_data)

# Combine imagined data files
imagined_data = load_and_concatenate(imagined_files, data_dir)
np.save(os.path.join(data_dir, 'imagined_data_fists.npy'), imagined_data)

print("Data files combined and saved successfully.")