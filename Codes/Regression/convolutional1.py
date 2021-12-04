from skimage.io import imread
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
import math
import pandas as pd
import os

# Create a list of epochs
epochs = 150
batch_size = 128
title = f"6E+S_Convolution1_{epochs}_Epochs"
# path = os.path.dirname(os.path.dirname(os.getcwd()))
# data_dir = rf'{path}\Datasets\Small_E\Holors'
data_dir = r"/home/mloper23/Datasets/E+S/Holors"

df = pd.read_excel(r'/home/mloper23/Dataframes/HolorsTotal.xlsx')
# df = pd.read_excel(r'C:\Users\mlope\OneDrive - Universidad EAFIT\EAFIT\Autofocusing\Dataframes\HolorsTotal.xlsx')

# tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

x_set = os.listdir(f"{data_dir}/D1") + os.listdir(f"{data_dir}/D2") + os.listdir(f"{data_dir}/D3") + os.listdir(
    f"{data_dir}/D4") + os.listdir(f"{data_dir}/D5")
y_set = pd.DataFrame()
for name in (names.split('.')[0] for names in x_set):
    y_set = pd.concat([y_set, df[df['name'] == name]['distance']])
x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.2, random_state=42)


class LoadImagesTrain(tf.keras.utils.Sequence):
    def __init__(self, x, y, d_path, b_size):
        self.x, self.y = x, y
        self.batch_size = b_size
        self.dir = d_path

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            imread(f"{self.dir}/{file_name.split('_')[0]}/{file_name}")
            for file_name in batch_x]), np.array(batch_y)


class LoadImagesValidation(tf.keras.utils.Sequence):
    def __init__(self, x, y, d_path, b_size):
        self.x, self.y = x, y
        self.batch_size = b_size
        self.dir = d_path

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return (np.array([
            imread(f"{self.dir}/{file_name.split('_')[0]}/{file_name}")
            for file_name in batch_x]), np.array(batch_y))


gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
img_height = 1024
img_width = 1024

sequence_train = LoadImagesTrain(x_train, y_train, data_dir, batch_size)
sequence_val = LoadImagesValidation(x_test, y_test, data_dir, 32)

with strategy.scope():
    # Model creation
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 1)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])
    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['mae', tf.keras.metrics.CosineSimilarity(axis=1)])
# checkpoint_path = "Models/" + title + '/weights.{epoch:02d}-{val_loss:.2f}'
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
hist_file = 'Training_plots/' + title
if not os.path.exists(hist_file):
    os.makedirs(hist_file)
hist_csv_file2 = 'Training_plots/' + title + '/history_model_' + title + '_cb' + '.csv'
history_logger = tf.keras.callbacks.CSVLogger(hist_csv_file2, separator=",", append=True)
history = model.fit(sequence_train, epochs=epochs, validation_data=sequence_val,
                    # callbacks=[history_logger, cp_callback])
                    callbacks=[history_logger])
if not os.path.exists('Models/' + title):
    os.makedirs('Models/' + title)
model.save('Models/' + title + '/Model' + title + '.h5')
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'Training_plots/' + title + '/history_model' + title + '.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)