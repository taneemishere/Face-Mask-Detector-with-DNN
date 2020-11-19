import os
from imutils import paths
from tensorflow import keras
import sklearn
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras import layers
from keras import Model
import matplotlib.pyplot as plt

# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
INIT_LR = 1.0004
EPOCHS = 20
BATCH_SIZE = 32

DIRECTORY = r"C:\Users\Hp\Projects\Machine Learning\Face-Mask-Detector-with-DNN\data"
CATEGORIES = ['with_mask', 'without_mask']

print('loading images....')

data = []
label = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = keras.preprocessing.image.load_img(img_path, target_size=(244, 244))
        image = keras.preprocessing.image.img_to_array(image)
        image = keras.applications.mobilenet_v2.preprocess_input(image)

        data.append(image)
        label.append(category)

# perform one-hot encoding on the labels
label_binarizer = LabelBinarizer()
label = label_binarizer.fit_transform(label)
label = keras.utils.to_categorical(label)

# converting to numpy arrays
data = np.array(data, dtype='float32')
label = np.array(label)

(train_X, test_X, train_y, test_y) = sklearn.model_selection.train_test_split(
    data, label, test_size=0.20, stratify=label, random_state=42
)

# constructing the training image generator for data Augmentation

augmentation = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# load the MobileNetV2 network, ensuring the head fully connected layer sets are left off

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_tensor=Input(shape=(244, 244, 3))
)

# construct the head of the model that will be placed on top of the base model

head_model = base_model.output
head_model = layers.AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = layers.Flatten(name='flatten')(head_model)
head_model = layers.Dense(128, activation='relu')(head_model)
head_model = layers.Dropout(0.5)(head_model)
head_model = layers.Dense(2, activation='softmax')(head_model)

# place the head Fully Connected model on top of the base model (this will become the actual
# model that we will train)

model = Model(inputs=base_model.input, output=head_model)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process

for layer in base_model.layers:
    layer.trainable = False

print('model is  compiling.....')

optimizer = keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# train the head of the network
print('training the head of the network....')
H = model.fit(
    augmentation.flow(train_X, train_y, batch_size=BS),
    steps_per_epoch=len(train_X) // BS,
    validation_data=(test_X, test_y),
    validation_steps=len(test_X) // BS,
    epochs=EPOCHS
)

print('evaluating the network....')
predictions = model.predict(test_X, batch_size=BS)

# for each image in the test set we need to find the index of the
# label with corresponding largest predicted probability
predictions = np.argmax(predictions, axis=1)

# show nicely formatted classification report
print(sklearn.metrics.classification_report(
    test_y.argmax(axis=1),
    predictions,
    target_names=lb.classes_)
)

# serialize the model to disk
print('\nsaving face mask detector model....')
model.save('face-mask-detector.model', save_format='h5')

# plot the training loss and accuracy
N = EPOCHS
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arrange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arrange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arrange(0, N), H.history['accuracy'], label='train_acc')
plt.plot(np.arrange(0, N), H.history['val_accuracy'], label='val_acc')
plt.xlabel("No. of Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot_model.png")
