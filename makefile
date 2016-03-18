CC=g++
CFLAGS=`pkg-config opencv visionworks --cflags`
LDFLAGS=`pkg-config opencv visionworks --libs`

all: cpu_bm gpu_bm gpu_dbf gpu_asw cpu_asw cv_sgbm vx_sgbm vx_bm

vx_bm:vx_bm.cpp
	g++ $(CFLAGS) -o $@ $< $(LDFLAGS)

vx_sgbm:vx_sgbm.cpp
	g++ $(CFLAGS) -o $@ $< $(LDFLAGS)

cv_sgbm:cv_sgbm.cpp
	g++ $(CFLAGS) -o $@ $< $(LDFLAGS)

cpu_asw:cpu_asw.cpp
	g++ $(CFLAGS) -o $@ $< $(LDFLAGS)

gpu_asw:gpu_asw.cu
	nvcc $(CFLAGS) -o $@ $< $(LDFLAGS)

gpu_dbf:gpu_dbf.cpp
	g++ $(CFLAGS) -o $@ $< $(LDFLAGS)

gpu_bm:gpu_bm.cpp
	g++ $(CFLAGS) -o $@ $< $(LDFLAGS)

cpu_bm:cpu_bm.cpp
	g++ $(CFLAGS) -o $@ $< $(LDFLAGS)

run:
	./cpu_bm

clean:
	rm *.o cpu_bm gpu_bm gpu_dbf gpu_asw cpu_asw cv_sgbm
