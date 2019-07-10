# Face Recognition

# Install dlib and face_recognition on Raspberry Pi

## Steps

Install required libraries with commands:
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    python3-pip \
    zip
sudo apt-get clean
```

Install the picamera python library with array support
```
sudo apt-get install python3-camera
sudo pip3 isntall --upgrade picamera[array]
```

Temporarily enable a larger swap file size (so the dlib compile won't fail due to limited memory):
```
sudo nano /etc/dphys-swapfile

< change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=1024 and save / exit nano >

sudo /etc/init.d/dphys-swapfile restart
```

Download and install dlib v19.6:
```
mkdir -p dlib
git clone -b 'v19.6' --single-branch https://github.com/davisking/dlib.git dlib/
cd ./dlib
sudo python3 setup.py install --compiler-flags "-mfpu=neon"
```

Install `face_recognition`:
```
sudo pip3 install face_recognition
```

Revert the swap file size change now that dlib is installed:
```
sudo nano /etc/dphys-swapfile

< change CONF_SWAPSIZE=1024 to CONF_SWAPSIZE=100 and save / exit nano >

sudo /etc/init.d/dphys-swapfile restart
```

pipenv

```
git clone <url>
cd face_recog
sudo install pipenv
pipenv shell
pipenv install face_recognition
```
