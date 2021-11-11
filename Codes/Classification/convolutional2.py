import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pandas as pd
import os

batch_size = 32
epochs = 50
path = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = rf'{path}\Datasets\Small_E\Holors'
title = f"3SmallE_Convolution2_{epochs}_Epochs_PC"
# data_dir = r"/home/mloper23/Datasets/E+S/Holors"

# tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")


def dataset(dir_path, height, width, color, batch):
    train = tf.keras.preprocessing.image_dataset_from_directory(
        dir_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(height, width),
        batch_size=batch,
        color_mode=color)
    test = tf.keras.preprocessing.image_dataset_from_directory(
        dir_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch,
        color_mode=color)

    return train, test


img_height = 1024
img_width = 1024
num_classes = 5
train_ds, val_ds = dataset(data_dir, img_height, img_width, 'grayscale', batch_size)
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 1)),
        layers.Conv2D(64, 7, padding='same', activation='relu'),
        layers.experimental.preprocessing.Normalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 5, padding='same', activation='relu'),
        layers.experimental.preprocessing.Normalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 5, padding='same', activation='relu'),
        layers.experimental.preprocessing.Normalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.experimental.preprocessing.Normalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.experimental.preprocessing.Normalization(),
        layers.MaxPooling2D(),
        layers.Dropout(.15),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', 'mae', 'mse'], run_eagerly=True)
# checkpoint_path = "Models/" + title + '/weights.{epoch:02d}-{val_loss:.2f}'
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_best_only=True,
#                                                  verbose=1)
hist_file = 'Training_plots/' + title
if not os.path.exists(hist_file):
    os.makedirs(hist_file)
hist_csv_file2 = 'Training_plots/' + title + '/history_model_' + title + '_cb' + '.csv'
history_logger = tf.keras.callbacks.CSVLogger(hist_csv_file2, separator=",", append=True)
# history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[cp_callback, history_logger])
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[history_logger])
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'Training_plots/' + title + '/history_model' + title + '.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
if not os.path.exists('Models/' + title):
    os.makedirs('Models/' + title)
model.save('Models/' + title + '/Model' + title + '.h5')