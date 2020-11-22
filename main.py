import cv2
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils


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
