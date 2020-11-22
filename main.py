import cv2
from tensorflow.keras.models import load_model

# load the face detection model
prototxt_path = r"face_detector\deploy.prototxt"
weights_path = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
# read the network model with opencv
face_model = cv2.dnn.readNet(prototxt_path, weights_path)

# load the model which we saved
mask_model = load_model("face-mask-detector.h5")
