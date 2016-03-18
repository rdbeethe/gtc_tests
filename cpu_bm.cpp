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
	// read images in, grayscale style
	Mat im_l = imread("l.png",0);
	Mat im_r = imread("r.png",0);
	Mat disp(im_l.rows,im_l.cols,CV_16S);
	// this is more complicated than I think it ought to be
    Ptr<StereoBM> bm = StereoBM::create(16,9);

    // bm->setROI1(roi1);
    // bm->setROI2(roi2);
    bm->setPreFilterCap(31);
    bm->setBlockSize(9);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numberOfDisparities);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(15);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);

    bm->compute(im_l,im_r,disp);
    imshow("window",disp);
    waitKey(0);

	return 0;
}