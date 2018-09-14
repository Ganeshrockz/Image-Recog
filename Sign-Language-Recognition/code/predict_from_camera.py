#!/usr/bin/env python2
import os
import sys
import pyttsx
import traceback

import cv2
from sklearn.externals import joblib

from common.config import get_config
from common.image_transformation import apply_image_transformation
from common.image_transformation import resize_image


def get_image_from_label(label):
    testing_images_dir_path = get_config('testing_images_dir_path')
    image_path = os.path.join(testing_images_dir_path, label, '001.jpg')
    image = cv2.imread(image_path)
    return image


def main():
    model_name = sys.argv[1]
    if model_name not in ['svm', 'logistic', 'knn']:
        print("Invalid model-name '{}'!".format(model_name))
        return

    print("Using model '{}'...".format(model_name))

    model_serialized_path = get_config(
        "model_{}_serialized_path".format(model_name))
    print("Model deserialized from path '{}'".format(model_serialized_path))

    camera = cv2.VideoCapture(0)

    while True:

        ret, frame = camera.read()
        cv2.imshow("Recording",frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            if not ret:
                print("Failed to capture image!")
                continue
            frame = resize_image(frame, 400)
            r = cv2.selectROI(frame)
            imCrop = frame[int(r[1]):int(r[1]+r[3]) , int(r[0]):int(r[0]+r[2])]
            cv2.imshow("Image",imCrop)
            #cv2.imshow("Webcam recording", frame)
            frame = imCrop
            try:
                frame = apply_image_transformation(frame)
                frame_flattened = frame.flatten()
                classifier_model = joblib.load(model_serialized_path)
                predicted_labels = classifier_model.predict(frame_flattened)
                predicted_label = predicted_labels[0]
                print("Predicted label = {}".format(predicted_label))
                predicted_image = get_image_from_label(predicted_label)
                predicted_image = resize_image(predicted_image, 200)
                cv2.imshow("Prediction = '{}'".format(predicted_label), predicted_image)
                engine = pyttsx.init()
                engine.say("The predicted text is " + str(predicted_label))
                engine.runAndWait()
                engine.stop()
            except Exception:
                exception_traceback = traceback.format_exc()
                print("Error while applying image transformation with the following exception trace:\n{}".format(exception_traceback))
        
    cv2.destroyAllWindows()
    print "The program completed successfully !!"


if __name__ == '__main__':
    main()
