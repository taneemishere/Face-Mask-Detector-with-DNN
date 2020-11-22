import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils.video import VideoStream
import imutils
import numpy as np


def detect_and_predict(frame, face_model, mask_model):
    # grab the dimensions of the frame and then construct a blob from it
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (244, 244), 104.0, 177.0, 123.0
    )
    # pass the blob(binary form of frames) to the network and obtain the
    # the face detections
    face_model.setInput(blob)
    detection = face_model.forward()
    print('\nShape of detections: ', detection.shape)
    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locations = []
    predictions = []

    # iterate over the detections
    for i in range(0, detection.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detection[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the x, y coordinates of the bounding box for the object
            box = detection[0, 0, i, 3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype('int')

            # making sure the bounding boxes fall within the dimensions
            # of the frame
            start_x, start_y = (max(0, start_x), max(0, start_y))
            end_x, end_y = (min(width - 1, end_x), min(height - 1, end_y))

            # extract the face area, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[start_y:end_y, start_x:end_x]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (244, 244))
            face = img_to_array(face)
            face = preprocess_input(face)


# load the face detection model
prototxt_path = r"face_detector\deploy.prototxt"
weights_path = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
# read the network model with opencv
face_model = cv2.dnn.readNet(prototxt_path, weights_path)

# load the model which we saved
mask_model = load_model("face-mask-detector.h5")

# capture the webcam
print('\nVideo is started....')
webcam = VideoStream(src=0).start()

# read the video stream until it is on
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = webcam.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locations, predictions) = detect_and_predict(frame, face_model, mask_model)
