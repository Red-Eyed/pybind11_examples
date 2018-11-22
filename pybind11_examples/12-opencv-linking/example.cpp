#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int opencv_usage(){
    Mat img(500, 1000, CV_8UC3, Scalar(0,0, 100));

    imshow("Image from C++", img);
    waitKey(0);
    destroyAllWindows();

    return 0;
}


namespace py = pybind11;

PYBIND11_MODULE(example, m)
{
  m.doc() = "pybind11 example plugin";

  m.def("opencv_usage", &opencv_usage);
}
