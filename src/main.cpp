#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::VideoCapture captureDevice;

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

void exportMesh(cv::Mat Depth, cv::Mat Normals, cv::Mat texture) {
    
    /* writing obj for export */
    std::ofstream objFile, mtlFile;
    objFile.open("export.obj");
    
    int width = Depth.cols;
    int height = Depth.rows;
    
    /* vertices, normals, texture coords */
    objFile << "mtllib export.mtl" << std::endl;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            objFile << "v " << x << " " << y << " " << Depth.at<float>(cv::Point(x,y)) << std::endl;
            objFile << "vt " << x/(width-1.0f) << " " << (1.0f-y)/height << " " << "0.0" << std::endl;
            objFile << "vn " << (float) Normals.at<cv::Vec3b>(y,x)[0] << " " << (float) Normals.at<cv::Vec3b>(y,x)[1] << " " << (float) Normals.at<cv::Vec3b>(y,x)[2] << std::endl;
        }
    }
    
    /* faces */
    objFile << "usemtl picture" << std::endl;
    for (int y = 0; y < height-1; y++) {
        for (int x = 0; x < width-1; x++) {
            int f1 = x+y*width+1;
            int f2 = x+y*width+2;
            int f3 = x+(y+1)*width+1;
            int f4 = x+(y+1)*width+2;
            objFile << "f " << f1 << "/" << f1 << "/" << f1 << " ";
            objFile << f2 << "/" << f2 << "/" << f2 << " ";
            objFile << f3 << "/" << f3 << "/" << f3 << std::endl;
            objFile << "f " << f2 << "/" << f2 << "/" << f2 << " ";
            objFile << f4 << "/" << f4 << "/" << f4 << " ";
            objFile << f3 << "/" << f3 << "/" << f3 << std::endl;
        }
    }
    
    /* texture */
    cv::imwrite("export.jpg", texture);
    mtlFile.open("export.mtl");
    mtlFile << "newmtl picture" << std::endl;
    mtlFile << "map_Kd export.jpg" << std::endl;
    
    objFile.close();
    mtlFile.close();
}

cv::Mat imageMask(std::vector<cv::Mat> camImages) {
    
    assert(camImages.size() > 0);
    cv::Mat image = camImages[0].clone();
    int quarter = image.cols/4.0;
    int eighth = image.rows/8.0;

    cv::Mat result, bgModel, fgModel;
    cv::Rect area(quarter, eighth, 3*quarter, 7*eighth);
    
    /* grabcut expects rgb images */
    cv::cvtColor(image, image, CV_GRAY2BGR);
    cv::grabCut(image, result, area, bgModel, fgModel, 1, cv::GC_INIT_WITH_RECT);
    
    cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
    return result;
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

cv::Mat computeNormals(std::vector<cv::Mat> camImages, cv::Mat Mask = cv::Mat()) {
    
    int height = camImages[0].rows;
    int width = camImages[0].cols;
    int numImgs = camImages.size();
    
    /* populate A */
	cv::Mat A(height*width, numImgs, CV_32FC1, cv::Scalar::all(0));
    for (int k = 0; k < numImgs; k++) {
        int idx = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
				A.at<float>(idx++, k) = camImages[k].data[i*width+j] * sgn(Mask.at<uchar>(cv::Point(j, i)));
            }
        }
    }
    
	/* speeding up computation, SVD from A^TA instead of AA^T */
    cv::Mat U,S,Vt;
	cv::SVD::compute(A.t(), S, U, Vt, cv::SVD::MODIFY_A);
	cv::Mat EV = Vt.t();
    
    cv::Mat N(height, width, CV_8UC3, cv::Scalar::all(0));
    int idx = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

			if (Mask.at<uchar>(cv::Point(j, i)) == 0) {
				N.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
			} else {
				float rSxyz = 1.0f / sqrt(EV.at<float>(idx, 0)*EV.at<float>(idx, 0) +
					                      EV.at<float>(idx, 1)*EV.at<float>(idx, 1) +
					                      EV.at<float>(idx, 2)*EV.at<float>(idx, 2));

				/* V contains the eigenvectors of A^TA, which are as well the z,x,y components of the surface normals for each pixel	*/
				float sz = 128.0f + 127.0f * sgn(EV.at<float>(idx, 0)) * fabs(EV.at<float>(idx, 0)) * rSxyz;
				float sx = 128.0f + 127.0f * sgn(EV.at<float>(idx, 1)) * fabs(EV.at<float>(idx, 1)) * rSxyz;
				float sy = 128.0f + 127.0f * sgn(EV.at<float>(idx, 2)) * fabs(EV.at<float>(idx, 2)) * rSxyz;

				N.at<cv::Vec3b>(i, j) = cv::Vec3b(sx, sy, sz);
			}
            idx += 1;
        }
    }
    
    return N;
}

void updateHeights(cv::Mat &Normals, cv::Mat &Z, int iterations) {
    
    for (int k = 0; k < iterations; k++) {
        for (int i = 1; i < Normals.rows-1; i++) {
            for (int j = 1; j < Normals.cols-1; j++) {
                float zU = Z.at<float>(cv::Point(j,i-1));
				float zD = Z.at<float>(cv::Point(j,i+1));
				float zL = Z.at<float>(cv::Point(j-1,i));
				float zR = Z.at<float>(cv::Point(j+1,i));
				float nxC = Normals.at<cv::Vec3b>(cv::Point(j,i))[0];
				float nyC = Normals.at<cv::Vec3b>(cv::Point(j,i))[1];
				float nxU = Normals.at<cv::Vec3b>(cv::Point(j,i-1))[0];
                float nyU = Normals.at<cv::Vec3b>(cv::Point(j,i-1))[1];
                float nxD = Normals.at<cv::Vec3b>(cv::Point(j,i+1))[0];
                float nyD = Normals.at<cv::Vec3b>(cv::Point(j,i+1))[1];
                float nxL = Normals.at<cv::Vec3b>(cv::Point(j-1,i))[0];
				float nyL = Normals.at<cv::Vec3b>(cv::Point(j-1,i))[1];
                float nxR = Normals.at<cv::Vec3b>(cv::Point(j+1,i))[0];
                float nyR = Normals.at<cv::Vec3b>(cv::Point(j+1,i))[1];
                int up = nxU == 0 && nyU == 0 ? 0 : 1;
                int down = nxD == 0 && nyD == 0 ? 0 : 1;
                int left = nxL == 0 && nyL == 0 ? 0 : 1;
                int right = nxR == 0 && nyR == 0 ? 0 : 1;
				if (up > 0 && down > 0 && left > 0 && right > 0)
					Z.at<float>(cv::Point(j,i)) = 1.0f/4.0f * ( zD + zU + zR + zL + nxU - nxC + nyL - nyC );
            }
        }
    }
}

cv::Mat cvtFloatToGrayscale(cv::Mat F, int limit = 255) {
    double min, max;
	cv::minMaxIdx(F, &min, &max);
	cv::Mat adjMap;
	cv::convertScaleAbs(F, adjMap, limit / max);
    return adjMap;
}

cv::Mat localHeightfield(cv::Mat Normals) {

    const int pyramidLevels = 4;
    const int iterations = 700;
    
    /* building image pyramid */
    std::vector<cv::Mat> pyrNormals;
    cv::Mat Normalmap = Normals.clone();
    pyrNormals.push_back(Normalmap);
    for (int i = 0; i < pyramidLevels; i++) {
        cv::pyrDown(Normalmap, Normalmap);
        pyrNormals.push_back(Normalmap.clone());
    }
    
    /* updating depth map along pyramid levels, starting with smallest level at top */
    cv::Mat Z(pyrNormals[pyramidLevels-1].rows, pyrNormals[pyramidLevels-1].cols, CV_32FC1, cv::Scalar::all(0));
    for (int i = pyramidLevels-1; i > 0; i--) {
        updateHeights(pyrNormals[i], Z, iterations);
        cv::pyrUp(Z, Z);
    }

    /* linear transformation of matrix values from [min,max] -> [a,b] */
    double min, max;
	cv::minMaxIdx(Z, &min, &max);
    double a = 0.0, b = 150.0;
    for (int i = 0; i < Normals.rows; i++) {
        for (int j = 0; j < Normals.cols; j++) {
            Z.at<float>(cv::Point(j,i)) = (float) a + (b-a) * ((Z.at<float>(cv::Point(j,i)) - min) / (max-min));
        }
    }
    
    return Z;
}

int main(int argc, char *argv[]) {
    
    int screenWidth = 1440;
    int screenHeight = 900;
    int numPics = 4;

	/* create capture device (webcam on Macbook Pro) */
	std::vector<cv::Mat> camImages;
	captureDevice = cv::VideoCapture(CV_CAP_ANY);
	if (!captureDevice.isOpened()) {
		std::cerr << "capture device error" << std::endl;
		/* using asset images */
		for (int i = 0; i < numPics; i++) {
            std::stringstream s;
            s << "../../images/image" << i << ".jpg";
            camImages.push_back(cv::imread(s.str(), CV_LOAD_IMAGE_GRAYSCALE));
            
		}
	} else {
		/* capture images from webcam while showing each pattern image */
		cv::namedWindow("camera", CV_WINDOW_NORMAL);
		for (int i=1; i<=numPics; i++) {
			cv::imshow("camera", lightPattern(screenWidth, screenHeight, i, numPics));
			cv::waitKey(0);
			cv::Mat frame;
			captureDevice >> frame;
			cv::cvtColor(frame, frame, CV_RGB2GRAY);
			camImages.push_back(frame.clone());
		}
        /* capture ambient image (dark pattern) */
        cv::imshow("camera", cv::Mat(screenHeight, screenWidth, CV_8UC1, cv::Scalar::all(0)));
        cv::waitKey(0);
        cv::Mat ambient;
        captureDevice >> ambient;
        cv::cvtColor(ambient, ambient, CV_RGB2GRAY);
        
        /* subtract cam images with ambient image */
        for (int i=0; i<camImages.size(); i++) {
            std::stringstream s;
            s << "camera_" << i << ".jpg";
            camImages[i] = camImages[i]-ambient;
            cv::imwrite(s.str(), camImages[i]);
        }
	}
    
    /* display images */
    for (int i=0; i<camImages.size(); i++) {
        std::stringstream s;
        s << "0" << i << ".jpg";
        cv::imshow(s.str(), camImages[i]);
    }

    
    /* threshold images */
    cv::Mat Mask = imageMask(camImages);
	cv::imshow("Mask", Mask);
    
    /* compute normal map */
    cv::Mat S = computeNormals(camImages, Mask);
	cv::Mat Normalmap;
	cv::cvtColor(S, Normalmap, CV_BGR2RGB);
	cv::imshow("normalmap.png", Normalmap);

    /* compute depth map */
	cv::Mat Depth = localHeightfield(S);
    cv::imshow("Local Depthmap", cvtFloatToGrayscale(Depth));
    
    exportMesh(Depth, S, camImages[0]);
	cv::waitKey(0);
    
    return 0;
}
