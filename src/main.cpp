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
#include <vtkRenderer.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <vtkTriangle.h>
#include <vtkPLYWriter.h>
#include <vtkWindowedSincPolyDataFilter.h>

cv::VideoCapture captureDevice;

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

void displayMesh(cv::Mat depthImage, cv::Mat texture) {
    
    /* creating visualization pipeline which basically looks like this:
     vtkPoints -> vtkPolyData -> vtkPolyDataMapper -> vtkActor -> vtkRenderer */
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyDataMapper> modelMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    vtkSmartPointer<vtkActor> modelActor = vtkSmartPointer<vtkActor>::New();
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkCellArray> vtkTriangles = vtkSmartPointer<vtkCellArray>::New();
    
    int height = depthImage.rows;
    int width = depthImage.cols;
    
    /* insert x,y,z coords and color information */
    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);
    colors->SetName("Colors");
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            points->InsertNextPoint(x, y, depthImage.at<float>(cv::Point(x,y)));
            colors->InsertNextTuple3(texture.at<uchar>(cv::Point(x,y)),
                                     texture.at<uchar>(cv::Point(x,y)),
                                     texture.at<uchar>(cv::Point(x,y)));
        }
    }
    
    /* setup the connectivity between grid points */
    vtkSmartPointer<vtkTriangle> triangle = vtkSmartPointer<vtkTriangle>::New();
    triangle->GetPointIds()->SetNumberOfIds(3);
    for (int i=0; i<height-1; i++) {
        for (int j=0; j<width-1; j++) {
            triangle->GetPointIds()->SetId(2, j+(i*width));
            triangle->GetPointIds()->SetId(1, (i+1)*width+j);
            triangle->GetPointIds()->SetId(0, j+(i*width)+1);
            vtkTriangles->InsertNextCell(triangle);
            triangle->GetPointIds()->SetId(2, (i+1)*width+j);
            triangle->GetPointIds()->SetId(1, (i+1)*width+j+1);
            triangle->GetPointIds()->SetId(0, j+(i*width)+1);
            vtkTriangles->InsertNextCell(triangle);
        }
    }
    polyData->SetPoints(points);
    polyData->SetPolys(vtkTriangles);
    polyData->GetPointData()->SetScalars(colors);
    
    /* mesh smoothing */
    vtkSmartPointer<vtkWindowedSincPolyDataFilter> smoother = vtkSmartPointer<vtkWindowedSincPolyDataFilter>::New();
    smoother->SetInputData(polyData);
    smoother->SetNumberOfIterations(15);
    smoother->BoundarySmoothingOff();
    smoother->FeatureEdgeSmoothingOff();
    smoother->SetFeatureAngle(120.0);
    smoother->SetPassBand(.001);
    smoother->NonManifoldSmoothingOn();
    smoother->NormalizeCoordinatesOn();
    smoother->Update();
    
    /* meshlab-ish background */
    modelMapper->SetInputConnection(smoother->GetOutputPort());
    renderer->SetBackground(.45, .45, .9);
    renderer->SetBackground2(.0, .0, .0);
    renderer->GradientBackgroundOn();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    modelActor->SetMapper(modelMapper);
    
    renderer->AddActor(modelActor);
    vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(renderWindow);

	/* exporting model */
	vtkSmartPointer<vtkPLYWriter> plyExporter = vtkSmartPointer<vtkPLYWriter>::New();
	plyExporter->SetInputData(polyData);
	plyExporter->SetFileName("export.ply");
	plyExporter->SetColorModeToDefault();
	plyExporter->SetArrayName("Colors");
	plyExporter->Update();
	plyExporter->Write();
    
    /* render mesh */
    renderWindow->Render();
    interactor->Start();
}

cv::Mat imageMask(std::vector<cv::Mat> camImages) {

	cv::Mat Max, Blurry, Mask;
	assert(camImages.size() >= 3);
	cv::GaussianBlur(camImages[0]+camImages[1]+camImages[2], Blurry, cv::Size(5,5), 2.0);
	cv::threshold(Blurry, Mask, 13, 255, CV_THRESH_BINARY);
	cv::dilate(Mask, Mask, cv::Mat());
    cv::dilate(Mask, Mask, cv::Mat());
    cv::erode(Mask, Mask, cv::Mat());
    cv::erode(Mask, Mask, cv::Mat());
    cv::erode(Mask, Mask, cv::Mat());
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
    
    /* populate A */
    cv::Mat A(height*width, numImgs, CV_32FC1);
    for (int k = 0; k < numImgs; k++) {
        int idx = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                A.at<float>(idx++, k) = camImages[k].data[i*width+j];
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
            float rSxyz = 1.0f / sqrt(EV.at<float>(idx, 0)*EV.at<float>(idx, 0) +
                                      EV.at<float>(idx, 1)*EV.at<float>(idx, 1) +
                                      EV.at<float>(idx, 2)*EV.at<float>(idx, 2));
            
            /* U contains the eigenvectors of AAT, which are as well the z,x,y components of the surface normals for each pixel	*/
            float sz = 128.0f + 127.0f * sgn(EV.at<float>(idx, 0)) * fabs(EV.at<float>(idx, 0)) * rSxyz;
            float sx = 128.0f + 127.0f * sgn(EV.at<float>(idx, 1)) * fabs(EV.at<float>(idx, 1)) * rSxyz;
            float sy = 128.0f + 127.0f * sgn(EV.at<float>(idx, 2)) * fabs(EV.at<float>(idx, 2)) * rSxyz;
            
            N.at<cv::Vec3b>(i, j) = cv::Vec3b(sz, sx, sy);
            idx += 1;
        }
    }
    
    return N;
}

cv::Mat localHeightfield(cv::Mat Normals, cv::Mat Mask) {

	int height = Normals.rows;
    int width = Normals.cols;
	cv::Mat Z(height, width, CV_32FC1, cv::Scalar::all(0));

	for (int k = 0; k < 5000; k++) {
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
                    if (Z.at<float>(cv::Point(j,i)) < 0.0f) {
                        Z.at<float>(cv::Point(j,i)) = 0.0f;
                    }
				}
			}
		}
	}

	double min, max;
	cv::minMaxIdx(Z, &min, &max);
	cv::Mat adjMap;
	cv::convertScaleAbs(Z, adjMap, 255 / max);
	cv::imshow("Local Depthmap", adjMap);

    /* linear transformation of matrix values from [min,max] -> [a,b] */
    double a = 0.0, b = 150.0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            Z.at<float>(cv::Point(j,i)) = (float) a + (b-a) * ((Z.at<float>(cv::Point(j,i)) - min) / (max-min));
        }
    }
    
    return Z;
}

cv::Mat globalHeightfield(cv::Mat Normals) {
    
    int height = Normals.rows;
    int width = Normals.cols;
	cv::Mat Pgrads(height, width, CV_32FC1, cv::Scalar::all(0));
    cv::Mat Qgrads(height, width, CV_32FC1, cv::Scalar::all(0));
    
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
			/* reordering surface normals (from U, which was z,x,y) */
			cv::Vec3f n = cv::Vec3f(Normals.data[y*width*3+x*3+1], Normals.data[y*width*3+x*3+2], Normals.data[y*width*3+x*3+0]);
			if (n[2] == 0.0f) { n[2] = 1.0f; }
			cv::normalize(n, n);
			Pgrads.at<float>(cv::Point(x,y)) = n[0]/n[2];
			Qgrads.at<float>(cv::Point(x,y)) = n[1]/n[2];
        }
    }

	cv::imshow("Pgrads", Pgrads);
	cv::imshow("Qgrads", Qgrads);

    cv::Mat P(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    cv::Mat Q(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    cv::Mat Z(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    
    /* p,q gradients from normal map */
    cv::dft(Pgrads, P, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(Qgrads, Q, cv::DFT_COMPLEX_OUTPUT);
    for (int i=0; i<Pgrads.rows; i++) {
        for (int j=0; j<Pgrads.cols; j++) {
            if (i != 0 || j != 0) {
				float u = sin(i*2.0f*float(CV_PI)/float(Pgrads.rows));
                float v = sin(j*2.0f*float(CV_PI)/float(Pgrads.cols));
                float uv = u*u + v*v;
                float d = uv;
                Z.at<cv::Vec2f>(i,j)[0] = (u*P.at<cv::Vec2f>(i,j)[1] + v*Q.at<cv::Vec2f>(i,j)[1]) / d;
                Z.at<cv::Vec2f>(i,j)[1] = (-u*P.at<cv::Vec2f>(i,j)[0] - v*Q.at<cv::Vec2f>(i,j)[0]) / d;
            }
        }
    }
    
    /* setting unknown average height to zero */
    Z.at<cv::Vec2f>(0,0)[0] = 0;
    Z.at<cv::Vec2f>(0,0)[1] = 0;
    
    cv::dft(Z, Z, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    /* display depth map */
	double min, max;
	cv::minMaxIdx(Z, &min, &max);
	std::cout << "min,max: " << min << "," << max << std::endl;
	cv::Mat adjMap;
	cv::convertScaleAbs(Z, adjMap, 255 / max);
	cv::imshow("Global Depthmap", adjMap);
    
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
    
    cv::Mat S = computeNormals(camImages);

	cv::Mat Normals(S.rows, S.cols, CV_8UC3, cv::Scalar::all(0));
	cv::Mat Mask = imageMask(camImages);
	S.copyTo(Normals, Mask);
	cv::imshow("(masked) normalmap.png", Normals);

	cv::Mat Depth = localHeightfield(Normals, Mask);
    double min, max;
    cv::minMaxIdx(Depth, &min, &max);
	displayMesh(Depth, camImages[0]);
	cv::waitKey(0);
    
    return 0;
}
