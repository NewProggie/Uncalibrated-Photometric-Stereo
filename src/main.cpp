#include <iostream>
#include <vector>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>

cv::VideoCapture captureDevice;

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

/**
 * Returns (binary) light pattern img L_j {j = 1..N} 
 */
cv::Mat lightPattern(int width, int height, int j, int N) {
    
    cv::Mat img(height, width, CV_8UC1, cv::Scalar::all(0));
    for (int y = -(height/2); y < height/2; y++) {
        for (int x = -(width/2); x < width/2; x++) {
            if (sgn(x*cv::cos(2*CV_PI*j/N)+y*cv::sin(2*CV_PI*j/N)) == 1) {
                img.at<uchar>(y+height/2,x+width/2) = 255;
            }
        }
    }
    
    return img;
}

cv::Mat computeNormals(std::vector<cv::Mat> camImages) {
    
    int height = camImages[0].rows;
    int width = camImages[0].cols;
    int numImgs = camImages.size();
    
    /* output matrix A = UDV */
    CvMat *U = cvCreateMat(height*width, numImgs, CV_32FC1);
    CvMat *A = cvCreateMat(height*width, numImgs, CV_32FC1);
    CvMat *D = cvCreateMat(numImgs, numImgs, CV_32FC1);
    CvMat *V = cvCreateMat(numImgs, numImgs, CV_32FC1);
    
    for (int k=0; k<numImgs; k++) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                A->data.fl[i*width*numImgs+j*numImgs+k] = (float) camImages[k].data[i*width+j];
            }
        }
    }
    
    /* cv::SVD::compute seems to be painfully slow in version 2.4.5 */
    cvSVD(A, D, U, V);
    cvReleaseMat(&A);
    cvReleaseMat(&V);
    cvReleaseMat(&D);
    
    cv::Mat S(height, width, CV_8UC3);
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            float rSxyz = 1 / sqrt(U->data.fl[i*width*numImgs+j*numImgs+0]*U->data.fl[i*width*numImgs+j*numImgs+0] +
                                   U->data.fl[i*width*numImgs+j*numImgs+1]*U->data.fl[i*width*numImgs+j*numImgs+1] +
                                   U->data.fl[i*width*numImgs+j*numImgs+2]*U->data.fl[i*width*numImgs+j*numImgs+2]);
            float sz = 128.0f+127.0f*sgn(U->data.fl[i*width*numImgs+j*numImgs+0])*fabs(U->data.fl[i*width*numImgs+j*numImgs+0])*rSxyz;
            float sx = 128.0f+127.0f*sgn(U->data.fl[i*width*numImgs+j*numImgs+1])*fabs(U->data.fl[i*width*numImgs+j*numImgs+1])*rSxyz;
            float sy = 128.0f+127.0f*sgn(U->data.fl[i*width*numImgs+j*numImgs+2])*fabs(U->data.fl[i*width*numImgs+j*numImgs+2])*rSxyz;
            
            S.data[i*width*3+j*3+0] = sz;
            S.data[i*width*3+j*3+1] = sx;
            S.data[i*width*3+j*3+2] = sy;
        }
    }

    cv::imwrite("normalmap.png", S);
    return S;
}

/**
 * Work in progress
 */
cv::Mat globalHeightfield(cv::Mat Normals) {
    
    cv::Mat Pgrads(Normals.rows, Normals.cols, CV_32F, cv::Scalar::all(0));
    cv::Mat Qgrads(Normals.rows, Normals.cols, CV_32F, cv::Scalar::all(0));
    
    for (int y=0; y<Normals.rows; y++) {
        for (int x=0; x<Normals.cols; x++) {
            float nz = Normals.data[y*Normals.cols*3+x*3+0];
            float nx = Normals.data[y*Normals.cols*3+x*3+1];
            float ny = Normals.data[y*Normals.cols*3+x*3+2];
            cv::Vec3f n(nx, ny, nz);
            cv::normalize(n, n);
            Pgrads.at<float>(cv::Point(x,y)) = n[0]/n[2];
            Qgrads.at<float>(cv::Point(x,y)) = n[1]/n[2];
        }
    }
    
    cv::Mat P(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    cv::Mat Q(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    cv::Mat Z(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    
    /* p,q gradients form normal map */
    cv::dft(Pgrads, P, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(Qgrads, Q, cv::DFT_COMPLEX_OUTPUT);
    for (int i=0; i<Pgrads.rows; i++) {
        for (int j=0; j<Pgrads.cols; j++) {
            if (i != 0 || j != 0) {
                float v = sin(i*2*CV_PI/Pgrads.rows);
                float u = sin(j*2*CV_PI/Pgrads.cols);
                float uv = u*u + v*v;
                float d = (1+0)*uv + 0*uv*uv;
                Z.at<cv::Vec2f>(i, j)[0] = (u*P.at<cv::Vec2f>(i, j)[1] + v*Q.at<cv::Vec2f>(i, j)[1]) / d;
                Z.at<cv::Vec2f>(i, j)[1] = (-u*P.at<cv::Vec2f>(i, j)[0] - v*Q.at<cv::Vec2f>(i, j)[0]) / d;
            }
        }
    }
    
    /* setting unknown average height to zero */
    Z.at<cv::Vec2f>(0, 0)[0] = 0;
    Z.at<cv::Vec2f>(0, 0)[1] = 0;
    
    cv::dft(Z, Z, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    
    return Z;
}

int main(int argc, char *argv[]) {

    /* create artificial light sources to put on monitor screen */
    std::vector<cv::Mat> binaryPatterns;
    for (int j=1; j<=4; j++) {
        binaryPatterns.push_back(lightPattern(1440, 900, j, 4));
    }
    
    /* create capture device (webcam on Macbook Pro) */
    captureDevice = cv::VideoCapture(CV_CAP_ANY);
    if (!captureDevice.isOpened()) {
        std::cerr << "capture device error" << std::endl;
        return -1;
    }
    
    /* capture images from webcam while showing each pattern image */
    std::vector<cv::Mat> camImages;
    cv::namedWindow("camera", CV_WINDOW_NORMAL);
    for (int i=0; i<binaryPatterns.size(); i++) {
        cv::imshow("camera", binaryPatterns[i]);
        cv::waitKey(0);
        
        cv::Mat frame;
        captureDevice >> frame;
        cv::cvtColor(frame, frame, CV_RGB2GRAY);
        camImages.push_back(frame.clone());
    }
    
    /* write images to disk */
    for (int i=0; i<camImages.size(); i++) {
        std::stringstream s;
        s << "0" << i << ".jpg";
        cv::imwrite(s.str(), camImages[i]);
    }
    
    cv::Mat S = computeNormals(camImages);
    globalHeightfield(S);
    
    return 0;
}
