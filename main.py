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

print('########################')
"""

Running face recognition on Live video from webcam
    - Process each video frame at 1/4 resolution to be more faster
    - Only detect faces in every other frame of video

"""


# argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-t', '--train', type = str, default = 'not', choices = ['yes','not'])
ap.add_argument('-pn', '--person_name', type = str, help = 'Person name to colect and train')
ap.add_argument('-e', '--encodings', required=True, help = 'path to serialized db of facial encodings')
ap.add_argument('-d', '--detection_method', type=str, default='cnn', help = 'face detection model to use hog or cnn')
ap.add_argument('-f', '--media', type=str, default='img', choices = ['img','vid'])
args = vars(ap.parse_args())

detection_method = args['detection_method']
encodings_pickle = args['encodings']
media = args['media']

# If train
if args['train'] == 'yes':
    person_name = args['person_name'].replace("_"," ")
    print('[INFO] - You choose train before start application')

    # define params
    capture = Capture()
    capture.create_dir_train(person_name)

    capture_ok = capture.capture_frames(media)

    if capture_ok:
        train = Train()
        train.processing(detection_method, encodings_pickle)
    #print(capture_ok)
    #train.processing()

#print('quit')
#quit()

# load pre-trained model
print('[INFO] - Loading encodings...')
data = pickle.loads(open(args['encodings'],'rb').read())

# webcam
video_capture = cv2.VideoCapture(0)

# Initialize variables
names = []
process_this_frame = True
# output dataframe
output = pd.DataFrame(columns=['Name','Timestamp'])
print(output)

while True:
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        """Find all the faces and face encodings in the current frame of video
        """

        faces_location = face_recognition.face_locations(rgb_small_frame, model=args['detection_method'])
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
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
