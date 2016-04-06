# GTC Talk: Bilateral and Trilateral Adaptive Support Weights in Stereo Vision

This code was used in the timing measurements of algorithms for a talk at the 2016 Nvidia GPU Technology Conference.

The code is also useful for testing a suite of various freely-available algorithms on your particular stereo images.

The version of my own GPU-accelerated adaptive support weight stereo matching algorithm was taken from a larger repository found at:

[https://github.com/rdbeethe/asw](https://github.com/rdbeethe/asw)

## Build instructions

If you have OpenCV 2.4.x and Nvidia VisionWorks and CUDA installed on your computer, you can just run

```
make
```

and

```
make run
```

Note that the cpu\_asw algorithm is very slow and is not tested by default.  You will have to execute it manually.

If you do not have Nvidia VisionWorks installed, you will have to adapt the makefile by removing the "visionworks" flag from the `pkg-config` commands, and modify the `all` rule to not depend on `vx_sgbm` and `vx_bm`.  Because the code was meant to be tested on Linux4Tegra (such as on the Jetson TK1 and TX1), it does not support OpenCV 3.0.x.

## Troubleshooting

If during your build you have CUDA installed but get errors such as "nvcc: command not found", you may have to modify your PATH environment variable to include "/usr/local/cuda<your cuda version>/bin".

If at run time you get an error sort of like "unable to find lib<cuda-something>.so", you may have to modify your LD\_LIBRARY\_PATH environment variableto include "/usr/local/cuda/lib" or "/usr/local/cuda/lib64", as appropriate.

