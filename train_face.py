import numpy as np
import cv2
import argparse
import os
from imutils import paths
import face_recognition
import pickle
import os
import imutils
from data_aquisition import DataAquisition
#from scipy import ndimage

class Capture():
    '''Create a folder with user name and capture frames to train
    '''

    def create_dir_train(self, person_name):
        '''Create a folder with user name
        '''

        self.person_name = person_name
        print('[INFO] - Create user: {}'.format(self.person_name))

        if not os.path.exists('./img/{}'.format(self.person_name)):
            os.makedirs('./img/{}'.format(self.person_name))


    def capture_frames(self,media):
        '''Open webcam and save X frames into directory created before
        '''

        if media == 'webcam':
            data_aq = DataAquisition(media)
            data_aq.initialize(0)
            status, frame = data_aq.get()
            #cap = cv2.VideoCapture(0)
            frame_i = 0

            while(True):
                status, frame = data_aq.get()
                img = frame

                if frame_i % 2 == 0 or frame_i % 5 == 0:
                    cv2.imwrite('./img/{}/{}.jpg'.format(self.person_name, frame_i), img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_i += 1
                cv2.imshow('frame', img)

                # limit to 60 frames to train

                if frame_i == 150:
                    break

        else:
            vidcap = cv2.VideoCapture('./vid/video.mp4')
            success, image = vidcap.read()

            count = 0

            while success:
                if count % 2 == 0 or count % 5 == 0:
                    cv2.imwrite('./img/{}/{}.jpg'.format(self.person_name, frame_i), img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_i += 1
                cv2.imshow('frame', img)

        cap.release()
        cv2.destroyAllWindows()

        capture_ok = True

        return capture_ok


class Train():
    '''Train model with those frames
    '''

    def processing(self, detection_method, encodings_pickle):
        print('traaaaaaaaaaaaaaaaaaaaaaain')
        self.detection_method = detection_method
        self.encodings_pickle = encodings_pickle
        self.imagePaths = list(paths.list_images('./img'))
        self.known_encodings = []
        self.known_names = []

        print('[INFO] - Quantifying faces...')
        known_encodings = self.known_encodings
        known_names = self.known_names

        for (i, imagePath) in enumerate(self.imagePaths):
            print('[INFO] - Processing image {}/{}'.format(i+1, len(self.imagePaths)))
            name = imagePath.split(os.path.sep)[-2]

            # BGR to RGB
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # detect coordinates of the bounding boxes
            boxes = face_recognition.face_locations(rgb, model=self.detection_method)

            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)

            # loop over the encodings:
            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(name)

        print('[INFO] - Serializing encodings...')
        data = {'encodings': known_encodings, 'names': known_names}
        print(data)
        print(self.encodings_pickle)
        f = open(self.encodings_pickle, 'wb')
        f.write(pickle.dumps(data))
        f.close()
