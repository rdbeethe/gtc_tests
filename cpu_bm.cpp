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
		printf("%s:%ds %dns\n",str,diffsec,diffnsec);
	}
	return (struct timespec) {diffsec, diffnsec};
}
 
int main(){
	// declare a time struct
	struct timespec ts;
	// read images in, grayscale style
	Mat im_l = imread("l.png",0);
	Mat im_r = imread("r.png",0);
	// disp image, and displayable disp image
	Mat disp(im_l.rows,im_l.cols,CV_16S);
	Mat disp8(im_l.rows,im_l.cols,CV_8U);

	// some constants
	int ndisp = 64;
	int sad_size = 21;
	// initialize the block matcher
    StereoBM bm(0,ndisp,sad_size);

	// start timer
	check_timer(NULL, &ts);
	// run the matcher
	bm(im_l,im_r,disp);
	// check timer
	check_timer("Time for cpu_bm", &ts);

	// convert image to be displayable
	disp.convertTo(disp8, CV_8U, 255/(ndisp*16.));
	//show image
    imshow("window",disp8);
    waitKey(0);

	return 0;
}
