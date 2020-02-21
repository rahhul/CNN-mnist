import numpy as np
import tensorflow as tf
import sklearn
import mnist
from tensorflow.keras.datasets import mnist

print(sklearn.__version__)
print(tf.__version__)

(train_images, train_labels), \
    (test_images, test_labels) = mnist.load_data()

print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)


img_rows, img_cols = 28, 28
channels = 1
input_shape = img_rows, img_cols, channels
num_classes = 10
batch_size = 64
epochs = 12


# Add channel dimension
train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)

print(train_images.shape)
print(test_images.shape)

# Convert dtypes to float32 and scale images to 0 and 1
train_images = train_images.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.

# Convert labels to categorical
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

# Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Conv2D(16, kernel_size=(3, 3), input_shape=[28, 28, 1], activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(32, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.2),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(num_classes, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam()

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())

print("Begin training...")

history = model.fit(train_images, train_labels, epochs=epochs,
                    validation_split=0.1, batch_size=batch_size)

score = model.evaluate(test_images, test_labels)

print("Test Loss: ", score[0])
print("Test Accuracy: ", score[1])