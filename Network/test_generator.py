import tensorflow as tf
from tensorflow import keras
import numpy as np


n_data = 1030

# Read CSV files from List
parameter = np.zeros((n_data, 5))

for i in range((n_data - 1) // 1000 + 1):
    parameter[1000 * i : min(1000 * (i + 1), n_data)] = np.genfromtxt("/hpcwork/cg457676/data/processed_parameter/pro_par{}.csv".format(i), delimiter = ",")[: min(1000, n_data - i * 1000)]

print(parameter)
print(parameter.shape)


# class SpectrogramDataGenerator(keras.utils.Sequence):
#     def __init__(self, csv_file_list, input_shape, num_classes, batch_size = 32, split='train', shuffle = False):

#         self.csv_file_list = csv_file_list
#         self.batch_size = batch_size
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.split = split  # 'train', 'validation', or 'test'
#         self.shuffle = shuffle
        
#         self.indices = np.arange(len(self.csv_file_list))

#         if self.shuffle:
#             np.random.shuffle(self.indices)
        
#         # Partition in train, validation and test data sets
#         if self.split == 'train':
#             self.indices = self.indices[:int(0.8 * len(self.indices))]
#         elif self.split == 'validation':
#             self.indices = self.indices[int(0.8 * len(self.indices)) : int(0.9 * len(self.indices))]
#         elif self.split == 'test':
#             self.indices = self.indices[int(0.9 * len(self.indices)):]
    
#     def __len__(self):
#         return int(np.ceil(len(self.csv_file_list) / self.batch_size))
    
#     def __getitem__(self, index):
#         start = index * self.batch_size
#         end = (index + 1) * self.batch_size
#         batch_indices = self.indices[start:end]
        
#         X = np.empty((len(batch_indices), *self.input_shape))
#         y = np.empty((len(batch_indices), self.num_classes))
        
#         for i, idx in enumerate(batch_indices):
#             csv_file_path = self.csv_file_list[idx]
#             data = np.genfromtxt(csv_file_path, delimiter = ",")  # Load your spectrogram data from the CSV file
#             label = parameter[idx]
            

#             print(csv_file_path)
#             print(idx)
            
#             # Preprocess and reshape your data if needed
#             # For example: spectrogram_data = preprocess(data.values)
#             #              X[i,] = spectrogram_data.reshape(self.input_shape)
            
#             # Assuming the label is in the last column of the CSV file
#             print(data.shape)
#             X[i] = data
            
#             y[i] = label
            
#         return X, y
    
#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)

# files = ["/hpcwork/cg457676/data/Processed_Data_0/" + "pspec0_{:05}.csv".format(i) for i in range(10000)]

# batch_size = 32
# input_shape = (79, 2001)
# num_classes = 5

# train_generator = SpectrogramDataGenerator(
#     csv_file_list = files,
#     batch_size = batch_size,
#     input_shape = input_shape,
#     num_classes = num_classes,
#     split = 'train'
# )

# valid_generator = SpectrogramDataGenerator(
#     csv_file_list = files,
#     batch_size = batch_size,
#     input_shape = input_shape,
#     num_classes = num_classes,
#     split = 'validation'
# )

# test_generator = SpectrogramDataGenerator(
#     csv_file_list = files,
#     batch_size = batch_size,
#     input_shape = input_shape,
#     num_classes = num_classes,
#     split = 'test'
# )

