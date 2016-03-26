#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <stdlib.h>
#include <time.h>

#define SIZE cones_l.rows,cones_l.cols

using namespace std;
using namespace cv;

// helper function for measuring time
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
	// declare timer
	struct timespec ts;
	// define the matcher
	int ndisp = 64;
	int winSize = 21;
	gpu::StereoBM_GPU bm(0,ndisp,winSize);
	// define the disparity bilateral filter
	gpu::DisparityBilateralFilter dbf; 
	dbf = gpu::DisparityBilateralFilter(30, 7, 3);
	// read in the images, grayscale
	Mat cones_l = imread("l.png",0);
	Mat cones_r = imread("r.png",0);
	// declare raw images
	gpu::GpuMat d_cones_l(SIZE,CV_8U), d_cones_r(SIZE,CV_8U);
	// declare disparity image
	gpu::GpuMat d_basic_disp(SIZE,CV_8U);
	// declare filtered disparity images
	gpu::GpuMat d_post_dbf(SIZE,CV_8U);
	Mat post_dbf(SIZE,CV_8U);

	// push the images to the GPU
	d_cones_l.upload(cones_l);
	d_cones_r.upload(cones_r);

	// do the basic match
	bm(d_cones_l,d_cones_r,d_basic_disp);

	// start the timer for just the dbf time
	check_timer(NULL, &ts);

	// do the disparity bilateral filter
	dbf(d_basic_disp,d_cones_l,d_post_dbf);

	// check gpu processing time
	check_timer("gpu_dbf time", &ts);

	// download results
	d_post_dbf.download(post_dbf);

	// show result
	imwrite("out/gpu_dbf.png",post_dbf*255/ndisp);
	//imshow("window",post_dbf);
	//waitKey(0);

	return 0;
}
