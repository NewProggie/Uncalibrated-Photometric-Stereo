#include <iostream>
#include <vector>
#include <sstream>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyData.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkImageViewer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>
#include <vtkRenderer.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <vtkTriangle.h>

cv::VideoCapture captureDevice;

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

void displayMesh(cv::Mat image) {
    
    /* creating visualization pipeline which basically looks like this:
     vtkPoints -> vtkPolyData -> vtkPolyDataMapper -> vtkActor -> vtkRenderer */
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyDataMapper> modelMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    vtkSmartPointer<vtkActor> modelActor = vtkSmartPointer<vtkActor>::New();
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkCellArray> vtkTriangles = vtkSmartPointer<vtkCellArray>::New();
    
    int height = image.rows;
    int width = image.cols;
    
    /* insert x,y,z coords */
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            points->InsertNextPoint(x, y, image.at<float>(y,x));
        }
    }
    
    /* setup the connectivity between grid points */
    vtkSmartPointer<vtkTriangle> triangle = vtkSmartPointer<vtkTriangle>::New();
    triangle->GetPointIds()->SetNumberOfIds(3);
    for (int i=0; i<height-1; i++) {
        for (int j=0; j<width-1; j++) {
            triangle->GetPointIds()->SetId(0, j+(i*width));
            triangle->GetPointIds()->SetId(1, (i+1)*width+j);
            triangle->GetPointIds()->SetId(2, j+(i*width)+1);
            vtkTriangles->InsertNextCell(triangle);
            triangle->GetPointIds()->SetId(0, (i+1)*width+j);
            triangle->GetPointIds()->SetId(1, (i+1)*width+j+1);
            triangle->GetPointIds()->SetId(2, j+(i*width)+1);
            vtkTriangles->InsertNextCell(triangle);
        }
    }
    polyData->SetPoints(points);
    polyData->SetPolys(vtkTriangles);
    
    /* create two lights */
    vtkSmartPointer<vtkLight> light1 = vtkSmartPointer<vtkLight>::New();
    light1->SetPosition(-1, 1, 1);
    renderer->AddLight(light1);
    vtkSmartPointer<vtkLight> light2 = vtkSmartPointer<vtkLight>::New();
    light2->SetPosition(1, -1, -1);
    renderer->AddLight(light2);
    
    /* meshlab-ish background */
    modelMapper->SetInputData(polyData);
    renderer->SetBackground(.45, .45, .9);
    renderer->SetBackground2(.0, .0, .0);
    renderer->GradientBackgroundOn();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    modelActor->SetMapper(modelMapper);
    
    /* setting some properties to make it look just right */
    modelActor->GetProperty()->SetSpecularColor(1, 1, 1);
    modelActor->GetProperty()->SetAmbient(0.2);
    modelActor->GetProperty()->SetDiffuse(0.2);
    modelActor->GetProperty()->SetInterpolationToPhong();
    modelActor->GetProperty()->SetSpecular(0.8);
    modelActor->GetProperty()->SetSpecularPower(8.0);
    
    renderer->AddActor(modelActor);
    vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(renderWindow);
    
    /* render mesh */
    renderWindow->Render();
    interactor->Start();
}

cv::Mat imageMask(std::vector<cv::Mat> camImages) {

	cv::Mat Max, Blurry, Mask;
	assert(camImages.size() >= 3);
	cv::GaussianBlur(camImages[0]+camImages[1]+camImages[2], Blurry, cv::Size(11,11), 2.5);
	cv::threshold(Blurry, Mask, 15, 255, CV_THRESH_BINARY);
	cv::dilate(Mask, Mask, cv::Mat());
	return Mask;
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
	cvSVD(A, D, U, V, CV_SVD_V_T);
    cvReleaseMat(&A);
    cvReleaseMat(&V);
    cvReleaseMat(&D);
    
	cv::Mat S(height, width, CV_8UC3, cv::Scalar::all(0));
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            float rSxyz = 1.0 / sqrt(U->data.fl[i*width*numImgs+j*numImgs+0]*U->data.fl[i*width*numImgs+j*numImgs+0] +
                                   U->data.fl[i*width*numImgs+j*numImgs+1]*U->data.fl[i*width*numImgs+j*numImgs+1] +
                                   U->data.fl[i*width*numImgs+j*numImgs+2]*U->data.fl[i*width*numImgs+j*numImgs+2]);
			/* U contains the eigenvectors of AAT, which are as well the z,x,y components of the surface normals for each pixel	*/
            float sz = 128.0f + 127.0f * sgn(U->data.fl[i*width*numImgs+j*numImgs+0]) * fabs(U->data.fl[i*width*numImgs+j*numImgs+0]) * rSxyz;
            float sx = 128.0f + 127.0f * sgn(U->data.fl[i*width*numImgs+j*numImgs+1]) * fabs(U->data.fl[i*width*numImgs+j*numImgs+1]) * rSxyz;
            float sy = 128.0f + 127.0f * sgn(U->data.fl[i*width*numImgs+j*numImgs+2]) * fabs(U->data.fl[i*width*numImgs+j*numImgs+2]) * rSxyz;
            
            S.data[i*width*3+j*3+0] = sz;
            S.data[i*width*3+j*3+1] = sx;
            S.data[i*width*3+j*3+2] = sy;
        }
    }

    return S;
}

cv::Mat localHeightfield(cv::Mat Normals, cv::Mat Mask) {

	int height = Normals.rows;
    int width = Normals.cols;
	cv::Mat Z(height, width, CV_32FC1, cv::Scalar::all(0));

	for (int k = 0; k < 1000; k++) {
		for (int i = 1; i < height-1; i++) {
			for (int j = 1; j < width-1; j++) {
				if (Mask.at<uchar>(cv::Point(j,i)) > 0) {
					int top = Mask.at<uchar>(cv::Point(j,i-1));
					int bottom = Mask.at<uchar>(cv::Point(j,i+1));
					int left = Mask.at<uchar>(cv::Point(j-1,i));
					int right = Mask.at<uchar>(cv::Point(j+1,i));
					float zBottom = Z.at<float>(cv::Point(j,i+1));
					float zTop = Z.at<float>(cv::Point(j,i-1));
					float zRight = Z.at<float>(cv::Point(j+1,i));
					float zLeft = Z.at<float>(cv::Point(j-1,i));
					float uTop = Normals.at<cv::Vec3b>(cv::Point(j,i-1))[1];
					float uCurr = Normals.at<cv::Vec3b>(cv::Point(j,i))[1];
					float uLeft = Normals.at<cv::Vec3b>(cv::Point(j-1,i))[2];
					float uRight = Normals.at<cv::Vec3b>(cv::Point(j,i))[2];
					if (top > 0 && bottom > 0 && left > 0 && right > 0) {
						Z.at<float>(cv::Point(j,i)) = 0.25 * (zBottom+zTop+zRight+zLeft+uTop-uCurr+uLeft-uRight);
					} else if (top > 0 && bottom == 0 && left > 0 && right > 0) {
						Z.at<float>(cv::Point(j,i)) = 0.33 * (zTop+zRight+zLeft)+0.25*(uTop-uCurr+uLeft-uRight);
					} else if (top == 0 && bottom > 0 && left > 0 && right > 0) {
						Z.at<float>(cv::Point(j,i)) = 0.33 * (zBottom+zRight+zLeft)+0.25*(-Normals.at<cv::Vec3b>(cv::Point(j,i+1))[1]+uCurr+uLeft-uRight);
					} else if (top > 0 && bottom > 0 && left == 0 && right > 0) {
						Z.at<float>(cv::Point(j,i)) = 0.33 * (zBottom+zTop+zRight)+0.25*(uTop-uCurr-Normals.at<cv::Vec3b>(cv::Point(j+1,i))[2]+uRight);
					} else if (top > 0 && bottom > 0 && left > 0 && right == 0) {
						Z.at<float>(cv::Point(j,i)) = 0.33 * (zBottom+zTop+zLeft)+0.25*(uTop-uCurr+uLeft-uRight);
					} else if (top > 0 && bottom == 0 && left == 0 && right > 0) {
						Z.at<float>(cv::Point(j,i)) = 0.5 * (zTop+zRight)+0.25*(uTop-uCurr-Normals.at<cv::Vec3b>(cv::Point(j+1,i))[2]+uRight);
					} else if (top > 0 && bottom == 0 && left > 0 && right == 0) {
						Z.at<float>(cv::Point(j,i)) = 0.5 * (zTop+zLeft)+0.25*(uTop-uCurr+uLeft-uRight);
					} else if (top == 0 && bottom > 0 && left == 0 && right > 0) {
						Z.at<float>(cv::Point(j,i)) = 0.5 * (zBottom+zRight)+0.25*(-Normals.at<cv::Vec3b>(cv::Point(j,i+1))[1]+uCurr-Normals.at<cv::Vec3b>(cv::Point(j+1,i))[2]+uRight);
					} else if (top == 0 && bottom > 0 && left > 0 && right == 0) {
						Z.at<float>(cv::Point(j,i)) = 0.5*(zBottom+zLeft)+0.25*(-Normals.at<cv::Vec3b>(cv::Point(j,i+1))[1]+uCurr+uLeft-uRight);
					}
				}
			}
		}
	}

	double min, max;
	cv::minMaxIdx(Z, &min, &max);
	cv::Mat adjMap;
	cv::convertScaleAbs(Z, adjMap, 100 / max);
	cv::imshow("Local Depthmap", adjMap);
	return adjMap;
}

cv::Mat globalHeightfield(cv::Mat Normals) {
    
	int height = Normals.rows;
    int width = Normals.cols;
    cv::Mat Pgrads(height, width, CV_32FC1, cv::Scalar::all(0));
    cv::Mat Qgrads(height, width, CV_32FC1, cv::Scalar::all(0));
    
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
			/* reordering surface normals (from U, which was zxy) */
			cv::Vec3b tmp = Normals.at<cv::Vec3b>(cv::Point(x,y));
			cv::Vec3f n = cv::Vec3f(float(tmp[1]), float(tmp[2]), float(tmp[0]));
			cv::normalize(n, n);

			/* offset: (row * numCols * numChannels) + (col * numChannels) + (channel) */
			Pgrads.at<float>(cv::Point(x,y)) = n[0]/n[2];
			Qgrads.at<float>(cv::Point(x,y)) = n[1]/n[2];
        }
    }

    cv::Mat P(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    cv::Mat Q(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    cv::Mat Z(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    
    /* p,q gradients from normal map */
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

	double min, max;
	cv::minMaxIdx(Z, &min, &max);
	cv::Mat adjMap;
	cv::convertScaleAbs(Z, adjMap, 100 / max);
	cv::imshow("Global Depthmap", adjMap);
    
    return Z;
}

int main(int argc, char *argv[]) {
    
	/* create capture device (webcam on Macbook Pro) */
	std::vector<cv::Mat> camImages;
	captureDevice = cv::VideoCapture(CV_CAP_ANY);
	if (!captureDevice.isOpened()) {
		std::cerr << "capture device error" << std::endl;
		/* using asset images */
		for (int i = 0; i < 3; i++)	{
			std::stringstream s;
			s << "..\\images\\0" << i << ".jpg";
			camImages.push_back(cv::imread(s.str(), CV_LOAD_IMAGE_GRAYSCALE));
		}
	} else {
		/* create artificial light sources to put on monitor screen */
		std::vector<cv::Mat> binaryPatterns;
		for (int j=1; j<=4; j++) {
			binaryPatterns.push_back(lightPattern(1440, 900, j, 4));
		}
		/* capture images from webcam while showing each pattern image */
		cv::namedWindow("camera", CV_WINDOW_NORMAL);
		for (int i=0; i<binaryPatterns.size(); i++) {
			cv::imshow("camera", binaryPatterns[i]);
			cv::waitKey(0);
			cv::Mat frame;
			captureDevice >> frame;
			cv::cvtColor(frame, frame, CV_RGB2GRAY);
			camImages.push_back(frame.clone());
		}
	}
    
    /* display images */
    for (int i=0; i<camImages.size(); i++) {
        std::stringstream s;
        s << "0" << i << ".jpg";
		cv::imshow(s.str(), camImages[i]);
    }
    
    cv::Mat S = computeNormals(camImages);
	cv::imshow("Normalmap", S);

	cv::Mat Normals(S.rows, S.cols, CV_8UC3, cv::Scalar::all(0));
	cv::Mat Mask = imageMask(camImages);
	S.copyTo(Normals, Mask);
	cv::imshow("Masked Normalmap", Normals);

	//cv::Mat Depth = localHeightfield(Normals, Mask);
	cv::Mat Depth = globalHeightfield(Normals);
	displayMesh(Depth);
	cv::waitKey(0);
    
    return 0;
}
