import imutils
import cv2
import os
import pickle
import numpy as np
os.chdir("/home/hrushikesh/images")
p = "/home/hrushikesh/images/rotated_images/"
contents = os.listdir(os.getcwd())
for (i,file) in enumerate(contents):
    if os.path.isfile(file):
        print("2")
        angle = np.random.choice([0, 90, 180, 270])
        image = cv2.imread(file)
        image = imutils.rotate_bound(image, angle)
        output_path = p + str(i) + ".jpg"
        cv2.imwrite(output_path , image) 
