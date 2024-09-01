import cv2 as cv
import numpy as np
import os
from time import time
# from captures.windowcapture import WindowCapture
from vision import Vision
import tensorflow as tf

cascade_face = cv.CascadeClassifier('cascade/cascade.xml')

# load an empty Vision class
vision_face = Vision(None)

# define a video capture object 
vid = cv.VideoCapture(0) 

loop_time = time()

emotion_model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(48, 48, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])
# Restore the weightsC:\Users\chris\OneDrive\Documents\Code\Projects\HaarCascasde Trainer\emotion_classifier\emotion_detection_model
# emotion_model.load_weights('emotion_detection_model/my_checkpoint.index')
# Create a checkpoint object
checkpoint = tf.train.Checkpoint(model=emotion_model)

# Restore the checkpoint
checkpoint.restore("emotion_detection_model/my_checkpoint").expect_partial()

while(True):
    
    #Get an updated image of a person
    # face_picture = cv.imread('positive/1(1).jpeg')
    # cv.imshow('Unprocessed', face_picture)
    
    # Capture the video frame 
    # by frame 
    ret, face_picture = vid.read() 
  
    # Display the resulting frame 
    # cv.imshow('frame', frame) 
    
    #Get rectangles
    rectangles = cascade_face.detectMultiScale(face_picture)
    
    if(len(rectangles) > 0):
        # Extract the region of interest (ROI) using the provided coordinates
        x, y, w, h = rectangles[0]
        x1, y1 = (x, y)
        x2, y2 = (x + w, y + h)
        roi = face_picture[y1:y2, x1:x2]
        # Get the dimensions of the ROI
        roi_height, roi_width = roi.shape[:2]

        # Create a black image with the same dimensions as the original
        black_image = np.zeros_like(face_picture)

        # Determine the top-right corner coordinates for placing the ROI
        image_height, image_width = face_picture.shape[:2]
        top_right_x = image_width - roi_width
        top_right_y = 0

        # Resize the image
        resized_image = cv.resize(roi, (48, 48))

        # Add the batch dimension
        input_data = np.expand_dims(resized_image, axis=0)

        # print(input_data.shape)  # This will print (1, 32, 48, 3)

        # print(roi.shape, resized_image.shape)
        print(emotion_model.predict(input_data)[0])

        # Overlay the ROI on the original image at the top-right corner
        # black_image[y1:y2, x1:x2] = roi #Working Code
        black_image[y1:y1 + 48, x1:x1 + 48] = resized_image
        
        cv.imshow('Processed', black_image)


    # # draw the detection results onto the original image
    # detection_image = vision_face.draw_rectangles(face_picture, rectangles)
    
    # cv.imshow('Processed', detection_image)

    # debug the loop rate
    # print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # press 'f' to save screenshot as a positive image, press 'd' to 
    # save as a negative image.
    # waits 1 ms every loop to process key presses
    key = cv.waitKey(1)
    if key == ord('q'):
        cv.destroyAllWindows()
        break
    # elif key == ord('f'):
    #     cv.imwrite('positive/{}.jpg'.format(loop_time), detection_image)
    # elif key == ord('d'):
    #     cv.imwrite('negative/{}.jpg'.format(loop_time), detection_image)

print('Done.')