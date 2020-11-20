import os
import numpy as np
from imutils import paths
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras import layers
from keras import Model
import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

INIT_LR = 1.0004
EPOCHS = 20
BATCH_SIZE = 32

# Our directory of dataset and the categories (labels)
DIRECTORY = r"C:\Users\Hp\Projects\Machine Learning\Face-Mask-Detector-with-DNN\data"
CATEGORIES = ['with_mask', 'without_mask']

print('Loading images....')

# List for data [features] and labels
data = []
label = []

# Getting the dataset
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = keras.preprocessing.image.load_img(img_path, target_size=(244, 244))
        image = keras.preprocessing.image.img_to_array(image)
        image = keras.applications.mobilenet_v2.preprocess_input(image)

        data.append(image)
        label.append(category)

# Perform one-hot encoding on the labels
print('\nPerforming Label Binarizer....')
label_binarizer = LabelBinarizer()
label = label_binarizer.fit_transform(label)
label = keras.utils.to_categorical(label)

# Converting data and labels to numpy arrays
print('\nConverting to numpy arrays....')
data = np.array(data, dtype='float32')
label = np.array(label)

# Split the data into training and testing sets
print('\nPerforming train test split....')
(train_X, test_X, train_y, test_y) = train_test_split(
    data, label, test_size=0.20, stratify=label, random_state=42
)

# Constructing the training image generator for Data Augmentation
"""
The data augmentation make some changes in the current images, like 
some changes in the pixel values and create a whole new image, or rotate it 
and many more.
"""
print('\nPerforming augmentation on Image data generator....')
augmentation = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load the MobileNetV2 network, ensuring the head fully connected
# layer sets are left off
print('\nBuilding base_model....')
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_tensor=layers.Input(shape=(244, 244, 3))
)

# Build the head of model that will be stacked on top of the base model
print('\nBuilding head_model....')
head_model = base_model.output
head_model = layers.AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = layers.Flatten(name='flatten')(head_model)
head_model = layers.Dense(128, activation='relu')(head_model)
head_model = layers.Dropout(0.5)(head_model)
head_model = layers.Dense(2, activation='softmax')(head_model)

# Place the head model (Fully Connected) on top of base model
# (this will become the actual model that we will train)
print('\nConnecting base and head model....')
model = Model(inputs=base_model.input, output=head_model)

# Iterate over all layers in the base model and freeze them so they WON'T
# be updated during the first training process
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
print('\nComiling the model....')
optimizer = keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the network
print('\nTraining the head of the network....')
H = model.fit(
    augmentation.flow(train_X, train_y, batch_size=BATCH_SIZE),
    steps_per_epoch=len(train_X) // BATCH_SIZE,
    validation_data=(test_X, test_y),
    validation_steps=len(test_X) // BATCH_SIZE,
    epochs=EPOCHS
)

# Do prediction of the test set
print('\nEvaluating the network....')
predictions = model.predict(test_X, batch_size=BATCH_SIZE)

# For each image in the test set we need to find the index of the
# label with corresponding largest predicted probability
predictions = np.argmax(predictions, axis=1)

# Print nicely formatted classification report of the model
print('\nModel report....\n')
print(sklearn.metrics.classification_report(
    test_y.argmax(axis=1),
    predictions,
    target_names=label_binarizer.classes_)
)

# Save the model for later use
print('\nsaving face mask detector model....')
model.save('face-mask-detector.h5')

# Plot the training loss and accuracy
print('\nMaking plots ready for model....')
N = EPOCHS
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, N), H.history['val_accuracy'], label='val_acc')
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot_model.png")
