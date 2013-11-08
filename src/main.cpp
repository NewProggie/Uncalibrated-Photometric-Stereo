#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char *argv[]) {

  /* reading images */
  std::vector<cv::Mat> images;
  images.push_back(cv::imread("../images/bunny1.png", CV_LOAD_IMAGE_GRAYSCALE));
  images.push_back(cv::imread("../images/bunny2.png", CV_LOAD_IMAGE_GRAYSCALE));
  images.push_back(cv::imread("../images/bunny3.png", CV_LOAD_IMAGE_GRAYSCALE));

  cv::waitKey(0);
  return 0;
}
