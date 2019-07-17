import face_recognition
import argparse
import pickle
import cv2
import numpy as np
import datetime
import pandas as pd
from datetime import date
from train_face import Capture
from train_face import Train
from data_aquisition import DataAquisition
import time

"""

Running face recognition on Live video from webcam
    - Process each video frame at 1/4 resolution to be more faster
    - Only detect faces in every other frame of video

"""


# argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-t', '--train', type = str, default = 'not', choices = ['only_train','not','grab_images','grab_and_train'])
ap.add_argument('-pn', '--person_name', type = str, help = 'Person name to colect and train')
ap.add_argument('-e', '--encodings', required=True, help = 'path to serialized db of facial encodings')
ap.add_argument('-d', '--detection_method', type=str, default='cnn', help = 'face detection model to use hog or cnn')
ap.add_argument('-m', '--media', type=str, default='webcam', choices = ['webcam','video','rasp','folder'])
ap.add_argument('-i', '--input', default = None)
ap.add_argument('-r', '--rotate', default = None, choices = ['0', '90', '180', '270'])
args = ap.parse_args()

#  I F   T R A I N   #

# GRAB
if args.train == 'grab_and_train' or args.train == 'grab_images':
    person_name = args.person_name.replace("_"," ")
    print('[INFO] - You choose train before start application')

    # capture frames from video or webcam
    capture = Capture()
    capture.create_dir_train(args.person_name)
    capture.capture_frames(args.media, args.person_name, args.input, args.rotate)

# TRAIN
if args.train == 'only_train' or args.train == 'grab_and_train':
    train = Train()
    train.processing(args.detection_method, args.encodings_pickle, args.person_name)

#  M O D E L  #

# load pre-trained model
print('[INFO] - Loading encodings...')
data = pickle.loads(open(args.encodings,'rb').read(),encoding='latin1')

data_aq = DataAquisition(args.media)
data_aq.initialize(0)

status, frame = data_aq.get()

# Initialize variables
names = []
process_this_frame = True
fps = []
output = pd.DataFrame(columns=['Name','Timestamp'])


while status:
    if process_this_frame:
        start = time.time()
        status, frame = data_aq.get()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        """Find all the faces and face encodings in the current frame of video
        """

        faces_location = face_recognition.face_locations(rgb_small_frame, model = args.detection_method)
        faces_encodings = face_recognition.face_encodings(rgb_small_frame, faces_location)

        faces_names = []

        for face_encoding in faces_encodings:
            """Is unknown or known
            """

            # Se if the face is a match for known face(s)
            matches = face_recognition.compare_faces(data['encodings'], face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(data['encodings'], face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = data['names'][best_match_index]

            faces_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(faces_location, faces_names):
        """Display Results and save output
        """

        # Scale back up face locations sincedd the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a label with a name below the face
        # If Unknown then red box
        # If known then green box
        if name == "Unknown":
            # Draw box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), cv2.FILLED)
        else:
            # Draw box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0,255,0), cv2.FILLED)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left +6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # output
        if name != "Unknown":
            output = output.append({'Name': name,
                                    'Timestamp': datetime.datetime.now()},
                                    ignore_index=True)

        # if 20h then save and reset dataset
        hour_to_save = '2000'
        hour_now = str(datetime.datetime.now().hour)+''+str(datetime.datetime.now().minute)

        if hour_to_save == hour_now:
            output.to_csv('./output/output-{}.csv'.format(str(date.today())), header = True, na_rep='', index=False)
            output = pd.DataFrame(columns=['Name','Timestamp'])


    # Display the resulting image
    cv2.namedWindow('Window', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Window', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # FPS
    fps.append(time.time() - start)
    print('[INFO] - FPS: {}'.format(1/np.mean(fps)))

cv2.destroyAllWindows()
