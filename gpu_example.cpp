#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <stdlib.h>
#include <time.h>

#define SIZE cones_l.rows,cones_l.cols,CV_8UC1
#define SIZE_SIGNED cones_l.rows,cones_l.cols,CV_8SC1
#define DISPSIZE cones_l.rows,cones_l.cols,CV_16SC1
#define OUTPUT_SCALE 4


 
using namespace cv;
using namespace std;

class disparity {
public:
	gpu::StereoBM_GPU bm;  // stereo matching object for disparity computation
	gpu::DisparityBilateralFilter dbf;
};

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
		printf("%s:%ds %dns\n",str,diffsec,diffnsec);
	}
	return (struct timespec) {diffsec, diffnsec};
}
 


int main(){
	// declare timer
	struct timespec timer;
	// define the matcher
	disparity matcher;
	matcher.bm.ndisp = 64;
	matcher.bm.winSize = 11;
	matcher.dbf = gpu::DisparityBilateralFilter(30, 3, 10);
	// read in the images, grayscale
	check_timer(NULL,&timer);
	Mat cones_l = imread("cones/im2.png",0);
	Mat cones_r = imread("cones/im6.png",0);
	check_timer("Time to read images",&timer);
	// declare raw images
	gpu::GpuMat d_cones_l(SIZE), d_cones_r(SIZE);
	// declare pre_bilateral filtered images
	Mat pre_bilateral_l(SIZE), pre_bilateral_r(SIZE), bg_sub_l(SIZE), bg_sub_r(SIZE);
	gpu::GpuMat d_pre_bilateral_l(SIZE), d_pre_bilateral_r(SIZE), d_bg_sub_l(SIZE), d_bg_sub_r(SIZE);
	// declare disparity images
	Mat basic_disp(SIZE), pre_bilateral_disp(SIZE), post_bilateral_disp(SIZE);
	gpu::GpuMat d_basic_disp(SIZE), d_pre_bilateral_disp(SIZE), d_post_bilateral_disp(SIZE);
	// declare temporary images
	Mat temp8s(SIZE_SIGNED);
	gpu::GpuMat d_temp8s(SIZE_SIGNED);

	// push the images to the GPU
	check_timer(NULL,&timer);
	d_cones_l.upload(cones_l);
	d_cones_r.upload(cones_r);
	check_timer("Time to upload images",&timer);

	// do the basic match
	printf("doing the basic match\n");
	check_timer(NULL,&timer);
	matcher.bm(d_cones_l,d_cones_r,d_basic_disp);
	check_timer("Time to do basic match of images",&timer);
	// download results
	check_timer(NULL,&timer);
	d_basic_disp.download(basic_disp);
	check_timer("Time to download result",&timer);
	// show result
	imshow("window",basic_disp*OUTPUT_SCALE);
	waitKey(0);
	// save image:
	imwrite("outputs/basic_disp.png",basic_disp*OUTPUT_SCALE);

	// do post-matching bilateral filter:
	printf("post-processing with bilateral fitler\n");
	check_timer(NULL,&timer);
	matcher.dbf(d_basic_disp,d_cones_l,d_post_bilateral_disp);
	check_timer("Time to do post-matching bilateral filter",&timer);
	// download results
	check_timer(NULL,&timer);
	d_post_bilateral_disp.download(post_bilateral_disp);
	check_timer("Time to download result",&timer);
	// show result
	imshow("window",post_bilateral_disp*OUTPUT_SCALE);
	waitKey(0);
	// save image:
	imwrite("outputs/post_bilateral_disp.png",post_bilateral_disp*OUTPUT_SCALE);

	// do pre-matching bilateral filters:
	printf("pre-processing with bilateral filter\n");
	check_timer(NULL,&timer);
	float sigma_color = 30;
	float sigma_spacial = 15;
	gpu::bilateralFilter(d_cones_l,d_pre_bilateral_l,15,sigma_color,sigma_spacial);
	gpu::bilateralFilter(d_cones_r,d_pre_bilateral_r,15,sigma_color,sigma_spacial);
	check_timer("Time to do pre-matching bilateral filters",&timer);
	// download results
	check_timer(NULL,&timer);
	d_pre_bilateral_l.download(pre_bilateral_l);
	check_timer("Time to download result",&timer);
	// show result
	imshow("window",pre_bilateral_l);
	waitKey(0);
	
	// save image:
	imwrite("outputs/pre_bilateral_l.png",pre_bilateral_l);

	// subtract background information (which is output of bilateral filter)
	check_timer(NULL,&timer);
	// to do the subtraction without underflow, subtract into signed 8-bit, then add 127
	gpu::subtract(d_cones_l,d_pre_bilateral_l,d_temp8s,gpu::GpuMat(),CV_8SC1);
	gpu::add(d_temp8s,127,d_bg_sub_l,gpu::GpuMat(),CV_8UC1);
	gpu::subtract(d_cones_r,d_pre_bilateral_r,d_temp8s,gpu::GpuMat(),CV_8SC1);
	gpu::add(d_temp8s,127,d_bg_sub_r,gpu::GpuMat(),CV_8UC1);
	check_timer("Time to subtract background",&timer);
	// download results
	check_timer(NULL,&timer);
	d_bg_sub_l.download(bg_sub_l);
	check_timer("Time to download result",&timer);
	// show result
	imshow("window",(bg_sub_l-70)*2);
	waitKey(0);
	// save image:
	imwrite("outputs/bg_sub_l.png",(bg_sub_l-70)*2);

	// compute disparity
	printf("computing disparity on bilaterally filtered images\n");
	check_timer(NULL,&timer);
	matcher.bm(d_bg_sub_l,d_bg_sub_r,d_pre_bilateral_disp);
	check_timer("Time to do basic match of images",&timer);
	// download results
	check_timer(NULL,&timer);
	d_pre_bilateral_disp.download(pre_bilateral_disp);
	check_timer("Time to download result",&timer);
	// show result
	imshow("window",pre_bilateral_disp*OUTPUT_SCALE);
	waitKey(0);
	// save image:
	imwrite("outputs/pre_bilateral_disp.png",pre_bilateral_disp*OUTPUT_SCALE);

	return 0;
}
