OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_cudaimgproc -lopencv_calib3d -lopencv_videostab -lopencv_objdetect -lopencv_video

all:
	g++ -x c++ kmeans_impl.cpp -std=c++11 $(OPENCV_LIBS)
