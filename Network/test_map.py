from tensorflow import keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['font.serif'] = []
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['errorbar.capsize'] = 2
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

#plt.rcParams['savefig.transparent'] = True
plt.rcParams['figure.figsize'] = (10, 6)

class SpectrogramDataGenerator(keras.utils.Sequence):
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
            self.indices = self.indices[int(0.8 * len(self.indices)) : int(0.9 * len(self.indices))]
        elif self.split == 'test':
            self.indices = self.indices[int(0.9 * len(self.indices)):]

        print(self.indices)
    
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
            data = np.reshape(np.genfromtxt(csv_file_path, delimiter = ","), (79, 2001, 1))  # Load your spectrogram data from the CSV file
            label = parameter[idx]
            

            # Assuming the label is in the last column of the CSV file

            X[i] = data
            y[i] = label
            
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Number of datasets with n_max = 1E4
n_data = 10

# Read CSV files (parameter and data sets)
parameter = np.zeros((n_data, 5))

for i in range((n_data - 1) // 1000 + 1):
    parameter[1000 * i : min(1000 * (i + 1), n_data)] = np.genfromtxt("/hpcwork/cg457676/data/processed_parameter/pro_par{}.csv".format(i), delimiter = ",")[: min(1000, n_data - i * 1000)]


files = ["/hpcwork/cg457676/data/Processed_Data_0/" + "pspec0_{:05}.csv".format(i) for i in range(n_data)]


# Generators for reading the data sets

batch_size = 32
input_shape = (79, 2001, 1)
num_classes = 5

train_generator = SpectrogramDataGenerator(
    csv_file_list = files,
    batch_size = batch_size,
    input_shape = input_shape,
    num_classes = num_classes,
    split = 'train'
)

valid_generator = SpectrogramDataGenerator(
    csv_file_list = files,
    batch_size = batch_size,
    input_shape = input_shape,
    num_classes = num_classes,
    split = 'validation'
)

test_generator = SpectrogramDataGenerator(
    csv_file_list = files,
    batch_size = batch_size,
    input_shape = input_shape,
    num_classes = num_classes,
    split = 'test'
)





# Creating the model of the CNN

model = keras.models.Sequential()
# model.add(keras.layers.Conv2D(16, (3, 3), activation = "relu", input_shape = (79, 2001, 1)))
# model.add(keras.layers.Conv2D(16, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((1,2)))

# model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
# model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((2,2)))

# model.add(keras.layers.Conv2D(10, (3, 3), activation = "relu"))
# model.add(keras.layers.Conv2D(10, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((2,2)))

# # Dense Layers

# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(32, activation = 'relu'))
# model.add(keras.layers.Dense(5, activation = "relu"))


# Test
model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (79, 2001, 1)))
model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
model.add(keras.layers.MaxPooling2D((1,2)))

model.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.Conv2D(16, (3, 3), activation = "relu"))
model.add(keras.layers.Conv2D(16, (3, 3), activation = "relu"))
model.add(keras.layers.MaxPooling2D((2,2)))

# Dense Layers

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation = 'relu'))
model.add(keras.layers.Dense(5, activation = "relu"))


model.summary()


model.compile(optimizer='adam',
              loss = "mean_absolute_percentage_error",
              metrics=['mean_absolute_percentage_error'])

history = model.fit(train_generator, epochs = 5, 
                    validation_data = valid_generator, verbose = 2)


# # save the model
# model.save("./run0_model.keras")

# # Save the history
# np.save('./run0_history.npy', history.history)
# # history = np.load("./my_first_model.keras", allow_pickle='TRUE').item()

fig, ax = plt.subplots()

ax.plot(history.history["loss"], label = "loss", color = "royalblue")
ax.plot(history.history["val_loss"], label = "val_loss", color = "crimson")
ax.legend()
ax.grid()

plt.savefig("./Network/run0_loss_plot.png")