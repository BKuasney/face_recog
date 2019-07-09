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
git clone
cd <directory>
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
python main.py --encodings encodings.pickle --train yes --person_name Power_Ranger --detection_method cnn
```

* To train and start on raspberry:
```
python main.py --encodings encodings.pickle --train yes --person_name Power_Ranger --detection_method hog
```

* To only start:
```
python main.py --encodings encodings.pickle --train no --detection_method cnn
```

* To only start on raspberry:
```
python main.py --encodings encodings.pickle --train no --detection_method hog
```
