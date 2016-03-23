#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>

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

int argmin_float(float* data, int len){
	float min = data[0];
	int idx = 0;
	for(int i=1; i<len; i++){
		if(data[i] < min){
			min = data[i];
			idx = i;
		}
	}
	return idx;
}

int asw(Mat im_l, Mat im_r, int ndisp, int s_sigma, int i_sigma){
	// window size and win_rad
	int win_rad = 1.5*s_sigma;
	int win_size = 2*s_sigma+1;

	// check that images are matching dimensions
	if(im_l.rows != im_r.rows){
		printf("Error: im_l and im_r do not have matching row count\n");
		return 1;
	}
	if(im_l.cols != im_r.cols){
		printf("Error: im_l and im_r do not have matching col count\n");
		return 1;
	}
	if(im_l.channels() != im_r.channels()){
		printf("Error: im_l and im_r do not have matching channel count\n");
		return 1;
	}

	// set easy-access variables for number of rows, cols, and chans
	int nrows = im_l.rows;
	int ncols = im_l.cols;
	int nchans = im_l.channels();
	// initialize the output data matrix
	unsigned char* out = (unsigned char*)malloc(nrows*ncols*sizeof(unsigned char));

	// get gaussian kernel for spacial look-up table:
	// equation from cv::getGaussianKernel(), but without normalization
	float s_weights[win_size][win_size]; 
	for(int i=0; i<win_size; i++){
		for(int j=0; j<win_size; j++){
			float x = i-win_rad;
			float y = j-win_rad;
			float radius = sqrt(x*x+y*y);
			s_weights[i][j] = std::pow(2.71828,-radius*radius/(2.*s_sigma*s_sigma));
			// printf("%.6f ",s_weights[i][j]);
		}
		// printf("\n");
	}

	// get gaussian kernel for intensity look-up table:
	// equation from cv::getGaussianKernel(), but without normalization
	float i_weights[511]; 
	for(int i=0; i<511; i++){
		float radius = i-255;
		i_weights[i] = std::pow(2.71828,-radius*radius/(2.*i_sigma*i_sigma));
		// printf("%.6f ",i_weights[i]);
	}

	// define a shortcut to the data arrays
	unsigned char* data_l = ((unsigned char*)(im_l.data));
	unsigned char* data_r = ((unsigned char*)(im_r.data));

	// start timer
	struct timespec ts;
	check_timer(NULL, &ts);

	// do asw
	// first two layers are to touch every pixel:
	for(int row = 0; row < nrows; row++){
		for(int col = 0; col < ncols; col++){
			// costs represnts the matching costs for this scanline:
			float* costs = (float*)malloc(ndisp*sizeof(float));
			// this layer is to scan along the search region:
			for(int disp = 0; disp < std::min(ndisp,col); disp++){
				// define floats for tracking this pixel's matching cost:
				float cost = 0;
				float normalizing_factor = 0;
				// get local pointers for l and r center pixels
				unsigned char* center_l = &data_l[(row*ncols + col)*nchans];
				// unsigned char* center_r = &data_r[(row*ncols + col - disp)*nchans];
				// the next two layers are to touch all neighborhood pixels:
				for(int j = std::max(0,row-win_rad); j < std::min(nrows,row+win_rad+1); j++){
					for(int i = std::max(disp,col-win_rad); i < std::min(ncols,col+win_rad+1); i++){
						// find the local variation coordinates
						int x = i - col;
						int y = j - row;
						// get a pointer to the variation pixel
						unsigned char* pixel_l = &data_l[( j * ncols + i ) * nchans];
						unsigned char* pixel_r = &data_r[( j * ncols + i - disp ) * nchans];
						// initialize the left and right weight with spacial sigma
						float weight_l = s_weights[x+win_rad][y+win_rad];
						float weight_r = weight_l;
						// also, initialize the sum of abs diff between windows
						int sad = 0;
						// this loop is to touch each channel
						for(int chan = 0; chan < nchans; chan++){
							//get the intensity abs difference for left and right
							int diff_l = abs(((int)pixel_l[chan]) - ((int)center_l[chan]));
							int diff_r = abs(((int)pixel_r[chan]) - ((int)center_l[chan]));
							// add the abs difference between the two windows
							sad += abs(((int)pixel_l[chan]) - ((int)pixel_r[chan]));
							// multiply in the weight from this channel
							weight_l *= i_weights[diff_l+255];
							weight_r *= i_weights[diff_r+255];
						}
						// trucate sad:
						// sad = std::min(sad,i_sigma);
						// we're done with this variation pixel:
						// add the weight times the difference to the cost
						cost += sad*weight_l*weight_r;
						normalizing_factor += weight_l*weight_r;
						// printf("normalizing_factor:%f\n",normalizing_factor);
					}
				}
				// add the cost for this window the list of costs
				costs[disp] = cost/normalizing_factor;
			}
			// set output image value
			out[row*ncols + col] = ((char)argmin_float(costs,ndisp));
			free(costs);
		}
	}

	// end timer
	check_timer("cpu_asw calculation time", &ts);

	//show the output matrix
	Mat outmat(nrows,ncols,CV_8UC1,out);
	imwrite("out/cpu_asw.png",outmat);
	// imshow("window",outmat);
	// waitKey(0);

	free(out);

	return 0;
}

int main(int argc, char** argv){
	// spacial and intensity sigmas
	int s_sigma = 4;
	int i_sigma = 30;
	// number of disparities to check
	int ndisp = 64;
	// input images
	Mat im_l = imread("l.png");
	Mat im_r = imread("r.png");

	return asw(im_l, im_r, ndisp, s_sigma, i_sigma);
}
