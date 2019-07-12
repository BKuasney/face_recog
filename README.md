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

## **USAGE**

On raspberry detection_method need to be 'hog'. Is more faster, but, less accurate.

* To train and start:
```
python main.py --encodings encodings.pickle --train yes --person_name Power_Ranger --detection_method cnn --media webcam
```
