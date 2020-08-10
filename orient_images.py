from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
sys.path.append("/home/hrushikesh/dl4cv/io")
from sklearn.preprocessing import LabelEncoder
import h5py
import cv2
import imutils
import pickle
from imutils import paths
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="path to HDF5 database")
# we need our hdf5 dataset only to extract the label names
ap.add_argument("-i", "--dataset", required=True, help="path to trained orientation model")
# path to the dataset of rotated images residing on disk
ap.add_argument("-m", "--model", required=True, help="path to trained orientation model")
args = vars(ap.parse_args())

# load the label names (i.e angles from the HDF5 datset
db = h5py.File(args["db"],'r')
labelNames = [int(angle) for angle in db["label_names"][:]]
db.close()

# grab the paths to the testing images and randomly sample 10 images from dataset
print("[INFO] sampling images...")
imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = np.random.choice(imagePaths, size=(12,), replace=False)

# load the VGG16 network
print("[INFO] loading network...")
vgg = VGG16(weights="imagenet", include_top=False)

# load the orientation model
print("[INFO] loading model...")
model = pickle.loads(open(args["model"], "rb").read())

# loop over the image paths
for imagePath in imagePaths:
    # load the image via OpenCV so we can manipulate it after classification
    orig = cv2.imread(imagePath)
    
    if orig.shape[0] > 800:
        continue
    # load the input image using the Keras helper utility while ensuring the image is resize to 224 * 224 pixels
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)

    # preprocess the image by (1) expanding the dimensions and (2) subtracting the mean RGB pixel intensity from the ImageNet dataset
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # pass the image through the network to obtain the feature vector
    features = vgg.predict(image)
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # now that we have the CNN features, pass these through our classifier to obtain the orientation predictions
    angle = model.predict(features)
    angle = labelNames[angle[0]]

    # now that we have the predicted orientation of the image, we can correct it
    rotated = imutils.rotate_bound(orig, 360 - angle)

    # display the original and corrected images
    cv2.imshow("Original", orig)
    print("Original image as predicted by model was rotated by: {} degrees".format(angle))
    cv2.imshow("Corrected", rotated)
    cv2.waitKey(0)


