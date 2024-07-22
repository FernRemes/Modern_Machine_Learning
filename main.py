import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from PIL import Image

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler


(x_train, y_train), (x_val, y_val) = cifar10.load_data()
x_train = x_train / 255
x_val = x_val / 255

y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train)
def lr_schedule(epoch):
        lr = 1e-3
        if epoch > 75:
            lr *= 0.5e-3
        elif epoch > 50:
            lr *= 1e-3
        elif epoch > 25:
            lr *= 1e-2
        return lr

lr_scheduler = LearningRateScheduler(lr_schedule)
# Train the model
model.fit(datagen.flow(x_train, y_train, batch_size=64),
            steps_per_epoch=x_train.shape[0] // 64,
            epochs=30,
            validation_data=(x_val, y_val),
            callbacks=[lr_scheduler])

# Save the model
model.save('cifar10.model.keras')