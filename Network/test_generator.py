import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class SpectrogramDataGenerator(keras.Sequence):
    def __init__(self, csv_file_list, input_shape, num_classes, batch_size = 32, split='train', shuffle = False):

        self.csv_file_list = csv_file_list
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.split = split  # 'train', 'validation', or 'test'
        self.shuffle = shuffle
        
        self.indices = np.arange(len(self.csv_file_list))

        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Partition in train, validation and test data sets
        if self.split == 'train':
            self.indices = self.indices[:int(0.8 * len(self.indices))]
        elif self.split == 'validation':
            self.indices = self.indices[int(0.8 * len(self.indices)):int(0.9 * len(self.indices))]
        elif self.split == 'test':
            self.indices = self.indices[int(0.9 * len(self.indices)):]
    
    def __len__(self):
        return int(np.ceil(len(self.csv_file_list) / self.batch_size))
    
    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_indices = self.indices[start:end]
        
        X = np.empty((len(batch_indices), *self.input_shape))
        y = np.empty((len(batch_indices), self.num_classes))
        
        for i, idx in enumerate(batch_indices):
            csv_file_path = self.csv_file_list[idx]
            data = pd.read_csv(csv_file_path)  # Load your spectrogram data from the CSV file
            
            # Preprocess and reshape your data if needed
            # For example: spectrogram_data = preprocess(data.values)
            #              X[i,] = spectrogram_data.reshape(self.input_shape)
            
            # Assuming the label is in the last column of the CSV file
            label = data.iloc[-1]
            y[i] = label
            
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


dir = "/hcpwork/cg457676/data/Processed_Data_0/"

trc = 0.8
vac = 0.1
tec = 0.1

files = [dir + "pspec{:05}.csv".format(i) for i in range(10000)]

train_int = int(len(files) * trc)
valid_int = int(len(files) * vac)
test_int = int(len(files) * tec)


train_files = files[:train_int]
valid_files = files[train_int : train_int + valid_int]
test_files = files[train_int + valid_int:]

