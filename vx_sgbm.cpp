#include <stdio.h>
#include <NVX/nvx.h>
#include <NVX/nvx_opencv_interop.hpp>
#include <opencv2/opencv.hpp>
#include <time.h>

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
		printf("%s:%ds %dns\n",str,diffsec,diffnsec);
	}
	return (struct timespec) {diffsec, diffnsec};
}


int main(){
	// declare the timer
	struct timespec ts;

	// read stereo images in as grayscale (CV_8U):
	cv::Mat mat_l = cv::imread("l.png",0);
	cv::Mat mat_r = cv::imread("r.png",0);
	if(mat_l.empty() || mat_r.empty()){
		printf("ERROR! failed to read one or both images, exiting...\n");
		return 1;
	}
	// read image size
	int w = mat_l.cols;
	int h = mat_l.rows;

	// create context, and graph
	vx_context c = vxCreateContext() ;
	vx_graph g = vxCreateGraph(c);

	/* create images; virtual images are for intermediate images that
	   aren't direct inputs or outputs from the graph, but due to the
	   simplicity of our graph, there are no intermediate images */
	vx_image left = nvx::createImageFromMat(c, mat_l);
	vx_image right = nvx::createImageFromMat(c, mat_r);
	vx_image disp = vxCreateImage (c , w, h, VX_DF_IMAGE_S16);
	// this is an example of a virtual image, if you needed one
	//vx_image intermed_im = vxCreateVirtualImage (g , 0, 0, VX_DF_IMAGE_VIRT);

	// values taken from demo installed by libvisionworks
	vx_int32 	minD = 0;
	vx_int32 	maxD = 64;
	vx_int32 	P1 = 8;
	vx_int32 	P2 = 109;
	vx_int32 	sad = 5;
	vx_int32 	clip = 31;
	vx_int32 	max_diff = 32000;
	vx_int32 	uniqueness = 0;

	// create the nodes in the graph
	vx_node n [] = {
		// there's only one node on this graph...
		nvxSemiGlobalMatchingNode(g, left, right, disp, minD, maxD, 
			P1, P2, sad, clip, max_diff, uniqueness, NVX_SCANLINE_ALL),
	};

	// verify then process the graph once
	printf("verifying graph now...\n");
	if(vxVerifyGraph(g) == VX_SUCCESS){
		printf("graph verified, processing now...\n");
		// start timer
		check_timer(NULL, &ts);
		// process graph
		int test = vxProcessGraph(g);
		// end timer
		check_timer("vx_sgbm processing time", &ts);
		printf("vxProcessGraph returns %d\n",test);
	}else{
		printf("graph verification failed, exiting...\n");
		vxReleaseGraph(&g);
		vxReleaseImage(&left);
		vxReleaseImage(&right);
		vxReleaseImage(&disp);
		vxReleaseContext(&c);
		return 0;
	}

	/* I'm using Nvidia's nvx_opencv_interop.hpp header file
	   to get a cv::Mat from a vx_image.  It seems there is also
	   something called NVXIO that looked more powerful/complicated */
	nvx::ImageToMatMapper mapper(disp);
	cv::Mat mat_d = mapper.getMat();

	// stop timer

	// to properly scale the image for viewing, we need min and max
	double minval;
	double maxval;
	cv::minMaxLoc(mat_d, &minval, &maxval);

	// now we need to scale from sint16 to uint8 before writing
	cv::Mat mat_im;
	mat_d.convertTo(mat_im, CV_8U, 255./(maxval-minval), -minval);
	// show image
	cv::imshow("window",mat_im);
	cv::waitKey(0);

	// I'm not completely sure that I'm releasing everything correctly
	vxReleaseGraph(&g);
	vxReleaseImage(&left);
	vxReleaseImage(&right);
	vxReleaseImage(&disp);
	vxReleaseContext(&c);

	return 0;
}
