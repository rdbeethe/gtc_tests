#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;
using namespace cv;

// helper function for checking timing of various parts of code
struct timespec check_timer(const char* str, struct timespec* ts){
	struct timespec oldtime;
	// copy old time over
	oldtime.tv_nsec = ts->tv_nsec;
	oldtime.tv_sec = ts->tv_sec;
	// update ts
	clock_gettime(CLOCK_REALTIME, ts);
	// print old time
	int diffsec;
	int diffnsec;
	if(str != NULL){
		diffsec =  ts->tv_sec - oldtime.tv_sec;
		diffnsec =  ts->tv_nsec - oldtime.tv_nsec;
		// correct the values if we measured over an integer second break:
		if(diffnsec < 0){
			diffsec--;
			diffnsec += 1000000000;
		}
		printf("%s:%ds %fms\n",str,diffsec,diffnsec/1e6);
	}
	return (struct timespec) {diffsec, diffnsec};
}
 
int main(){
	// declare a time struct
	struct timespec ts;
	// read images in, color style
	Mat im_l = imread("l.png");
	Mat im_r = imread("r.png");
	// disp image, and displayable disp image
	Mat disp(im_l.rows,im_l.cols,CV_16S);
	Mat disp8(im_l.rows,im_l.cols,CV_8U);

	// some constants
	int ndisp = 64;
	int sad_size = 3;
	int nchans = im_l.channels();
	// initialize the block matcher
    StereoSGBM sgbm(0,ndisp,sad_size);
	// set up the SG block matcher
	// (values from stereo_match.cpp example, from opencv library)
    sgbm.preFilterCap = 63;
	sgbm.P1 = 8*nchans*sad_size*sad_size;
	sgbm.P2 = 32*nchans*sad_size*sad_size;
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = 100;
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = 1;
	sgbm.fullDP = true;


	// start timer
	check_timer(NULL, &ts);
	// run the matcher
	sgbm(im_l,im_r,disp);
	// check timer
	check_timer("Time for cv_sgbm (cpu)", &ts);

	// convert image to be displayable
	disp.convertTo(disp8, CV_8U, 255/(ndisp*16.));
	//show image
	imwrite("out/cv_sgbm.png",disp8);
    //imshow("window",disp8);
    //waitKey(0);

	return 0;
}
