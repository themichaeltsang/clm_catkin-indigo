# clm_catkin-indigo

ROS wrapper for the [Cambridge face tracker (aka CLM framework)](https://github.com/TadasBaltrusaitis/CLM-framework). 
It works on catkin and it targets [ROS indigo](http://wiki.ros.org/indigo).

## Dependencies

### OpenCV > 3.0.0

#### Prerequisites

```
sudo apt-get -y install libopencv-dev build-essential cmake git libgtk2.0-dev pkg-config python-dev python-numpy libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff4-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils unzip
```

#### Download OpenCV

Download OpenCV from [opencv.org](opencv.org). I used the 3.1.0 version available [here](https://github.com/Itseez/opencv/archive/3.1.0.zip).

#### Install OpenCV

 * Unzip the archive in a suitable directory (we will be using `~/src/`)
 * `cd ~/src/opencv-3.1.0`
 * `mkdir build`
 * `cd build`
 * `cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON ..`
 * `make -j7`
 * `sudo make install`

#### Finish installation

To get OpenCV working properly, we need:

```
sudo /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
```