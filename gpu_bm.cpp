#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <stdlib.h>
#include <time.h>

#define SIZE cones_l.rows,cones_l.cols
#define OUTPUT_SCALE 4

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

	// some constants
	int ndisp = 64;
	int sad_size = 21;
	// initialize the block matcher
    gpu::StereoBM_GPU bm(0,ndisp,sad_size);

	// read in the images, grayscale
	Mat cones_l = imread("l.png",0);
	Mat cones_r = imread("r.png",0);
	// declare raw images
	gpu::GpuMat d_cones_l(SIZE,CV_8U), d_cones_r(SIZE,CV_8U);
	// declare disparity images
	Mat basic_disp(SIZE,CV_8U);
	gpu::GpuMat d_basic_disp(SIZE,CV_8U);

	// push the images to the GPU
	d_cones_l.upload(cones_l);
	d_cones_r.upload(cones_r);

	// start the timer
	check_timer(NULL, &ts);

	// do the basic match
	bm(d_cones_l,d_cones_r,d_basic_disp);

	// check processing time
	check_timer("gpu_bm", &ts);

	// download results
	d_basic_disp.download(basic_disp);
	
	// show result
	imwrite("out/gpu_bm.png",basic_disp*255/ndisp);
	//imshow("window",basic_disp*255/ndisp);
	//waitKey(0);

	return 0;
}
