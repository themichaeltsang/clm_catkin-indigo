# clm_catkin-indigo

ROS wrapper for the [Cambridge face tracker (aka CLM framework)](https://github.com/TadasBaltrusaitis/CLM-framework). 
It works on catkin and it targets [ROS indigo](http://wiki.ros.org/indigo).

## Dependencies

### 1 Prerequisites

```
sudo apt-get -y install libopencv-dev build-essential cmake git libgtk2.0-dev pkg-config python-dev python-numpy libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff4-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils unzip
```

### 2. OpenCV should be > 3.0.0

#### 2.1 Download OpenCV

Download OpenCV from [opencv.org](opencv.org). I used the 3.1.0 version available [here](https://github.com/Itseez/opencv/archive/3.1.0.zip).

#### 2.2 Install OpenCV

 * Unzip the archive in a suitable directory (we will be using `~/src/`)
 * `cd ~/src/opencv-3.1.0`
 * `mkdir build`
 * `cd build`
 * `cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON ..`
 * `make -j7`
 * `sudo make install`

#### 2.3 Finish installation

To get OpenCV working properly, we need:

```
sudo /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
```

### 3. CLM code should be manually added to the repository

This repository needs a couple libraries from the original CLM repo it is branching from. It would be nice to use `git submodule` routine, but considering that the CLM repo is more than `1GB` (not because of the repo itself, but because of its history that has been not managed over the time), it is better to directly download the minimum viable set of libraries.

#### 3.1 Download the code

The code is available [here](https://github.com/TadasBaltrusaitis/CLM-framework/tree/master/lib). You should copy `3rdParty/dlib` into `lib/3rdParty`. Then copy `local/CLM` and `local/FaceAnalyser` in `lib/local`.

#### 3.2 Modify `lib/local/CLM/CMakeLists.txt`

Replace the two `install` directives of the file `lib/local/CLM/CMakeLists.txt` with the following:

```
target_link_libraries(CLM opencv_calib3d opencv_objdetect)

## Mark library for installation
install(TARGETS CLM
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

## Mark cpp header files for installation
install(FILES ${HEADERS} 
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
```

#### 3.2 Modify `lib/local/FaceAnalyser/CMakeLists.txt`

Replace the two `install` directives of the file `lib/local/FaceAnalyser/CMakeLists.txt` with the following:

```
## Mark library for installation
install(TARGETS FaceAnalyser
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

## Mark cpp header files for installation
install(FILES ${HEADERS} 
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
```

## 4. Compilation and Installation

If everything is ok , simply `cd` in the catkin workspace, and type:

```
catkin_make
catkin_make install
rospack profile
```

## 5. Usage

