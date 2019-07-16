# Face Recognition


### Requirements

  * Python 3.3+ or Python 2.7
  * macOS or Linux
  * [dlib](https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65)
  * opencv (pip3 install opencv-python)


  [setup on raspberry pi] ()


## **SETUP**

### Setup Linux

```
git clone <URL>
cd <directory>
pip install git+https://github.com/ageitgey/face_recognition_models
pip install pipenv
pipenv shell
pipenv install face_recognition

```

### Setup Raspberry
[see](https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65)

## **USAGE CPU**

* To train with webcam and start webcam:
```
python3 main.py --encodings encodings.pickle --train yes --person_name Bruno_Kuasney --detection_method cnn --media webcam --input 0
```
* To only start webcam:
```
python3 main.py --encodings encodings.pickle --train not --detection_method cnn --media webcam --input 0
```

* To train with video and start webcam
```
python3 main.py --encodings encodings.pickle --train yes --detection_method cnn --media video --input ./vid/video.mp4
```

## **USAGE RASPBERRY PI 3**

On raspberry detection_method need to be 'hog'. Is more faster, but, less accurate.

* to train with webcam and start webcam [not recommended]:
```
python3 main.py --encodings encodings.pickle --train yes --person_name Bruno_Kuasney --detection_method hog --media webcam --input 0
```

* To only start webcam:
```
python3 main.py --encodings encodings.pickle --train not --detection_method hog --media webcam --input 0
```

* To train with video and start webcam [not recommended]
```
python3 main.py --encodings encodings.pickle --train yes --person_name Bruno Kuasney --detection_method hog --media video --input ./vid/video.mp4
```
