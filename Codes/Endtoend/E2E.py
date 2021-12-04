from skimage.io import imread
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input, experimental
from tensorflow.keras import Model
import numpy as np
import math
import pandas as pd
import os


epochs = 20
batch_size = 1
test_batch_size = 1
title = f"8Test_E(D1)_E2E_{epochs}_Epochs_{batch_size}_BatchSize"
path = os.path.dirname(os.path.dirname(os.getcwd()))
# data_dir = r'C:\Users\mlope\OneDrive - Universidad EAFIT\EAFIT\Autofocusing\Datasets\Test_E\Holors'
data_dir = r"/home/mloper23/Datasets/Test_E/Holors"
# data_real = r'C:\Users\mlope\OneDrive - Universidad EAFIT\EAFIT\Autofocusing\Datasets\Test_E\Reconstructions'
data_real = r"/home/mloper23/Datasets/Test_E/Reconstructions"
gpus = tf.config.list_physical_devices('GPU')
logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
# x_set = os.listdir(f"{data_dir}")
# x_set = os.listdir(f"{data_dir}/D1") + os.listdir(f"{data_dir}/D2") + os.listdir(f"{data_dir}/D3") + os.listdir(
#     f"{data_dir}/D4") + os.listdir(f"{data_dir}/D5")
x_set = os.listdir(f"{data_dir}/D1")
# y_set = os.listdir(f"{data_real}")
# y_set = os.listdir(f"{data_real}/D1") + os.listdir(f"{data_real}/D2") + os.listdir(f"{data_real}/D3") + os.listdir(
#     f"{data_real}/D4") + os.listdir(f"{data_real}/D5")
y_set = os.listdir(f"{data_real}/D1")
x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.2, random_state=42)


def build_model(input_layer, start_neurons):
    with strategy.scope():
        conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
        conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
        pool1 = Dropout(0.25)(pool1)

        conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
        conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
        pool2 = Dropout(0.5)(pool2)

        conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
        conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)
        pool3 = Dropout(0.5)(pool3)

        conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
        conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)
        pool4 = Dropout(0.5)(pool4)

        # Middle
        convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
        convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

        deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(0.5)(uconv4)
        uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
        uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

        deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Dropout(0.5)(uconv3)
        uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
        uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

        deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Dropout(0.5)(uconv2)
        uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
        uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

        deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Dropout(0.5)(uconv1)
        uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
        uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

        output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    return output_layer


class LoadImagesTrain(tf.keras.utils.Sequence):
    def __init__(self, x, y, d_path, b_size, r_path):
        self.x, self.y = x, y
        self.batch_size = b_size
        self.dir = d_path
        self.real = r_path

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.asarray([
            imread(f"{self.dir}/{file_name.split('_')[0]}/{file_name}")
            for file_name in batch_x]).astype(np.float32) / 255, np.asarray([
            imread(f"{self.real}/{file_name.split('_')[0]}/{file_name}")
            for file_name in batch_y]).astype(np.float32) / 255


class LoadImagesValidation(tf.keras.utils.Sequence):
    def __init__(self, x, y, d_path, b_size, r_path):
        self.x, self.y = x, y
        self.batch_size = b_size
        self.dir = d_path
        self.real = r_path

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.asarray([
            imread(f"{self.dir}/{file_name.split('_')[0]}/{file_name}")
            for file_name in batch_x]).astype(np.float32) / 255, np.asarray([
            imread(f"{self.real}/{file_name.split('_')[0]}/{file_name}")
            for file_name in batch_y]).astype(np.float32) / 255


gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
img_height = 1024
img_width = 1024

sequence_train = LoadImagesTrain(x_train, y_train, data_dir, batch_size, data_real)
sequence_val = LoadImagesValidation(x_test, y_test, data_dir, test_batch_size, data_real)

input_layer = Input((1024, 1024, 1))
# input_layer = Input((240, 240, 1))
output_layer = build_model(input_layer, 16)
with strategy.scope():
    unet = Model(input_layer, output_layer)
    unet.summary()
    unet.compile(optimizer="adam", loss="mse")
hist_file = 'Training_plots/' + title
# hist_file = r'C:\Users\mlope\OneDrive - Universidad EAFIT\EAFIT\Autofocusing/Training_plots/' + title
if not os.path.exists(hist_file):
    os.makedirs(hist_file)
hist_csv_file2 = 'Training_plots/' + title + '/history_model_' + title + '_cb' + '.csv'
# hist_csv_file2 = r'C:\Users\mlope\OneDrive - Universidad EAFIT\EAFIT\Autofocusing\Training_plots/' + title + '/history_model_' + title + '_cb' + '.csv'
history_logger = tf.keras.callbacks.CSVLogger(hist_csv_file2, separator=",", append=True)
history = unet.fit(sequence_train, epochs=epochs, validation_data=sequence_val, verbose=1,
                   # callbacks=[history_logger, cp_callback])
                   callbacks=[history_logger])
# model_path = r'C:\Users\mlope\OneDrive - Universidad EAFIT\EAFIT\Autofocusing\Training_plots/' + title
if not os.path.exists('Models/' + title):
    os.makedirs('Models/' + title)
unet.save('Models/' + title + '/Model' + title + '.h5')
# if not os.path.exists(model_path):
#     os.makedirs(model_path)
# unet.save(model_path + '/Model' + title + '.h5')
# hist_df = pd.DataFrame(history.history)
# hist_csv_file = 'Training_plots/' + title + '/history_model' + title + '.csv'
# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)
