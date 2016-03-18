CC=g++
CFLAGS=`pkg-config opencv --cflags`
LDFLAGS=`pkg-config opencv --libs`

all: cpu_bm gpu_bm

gpu_bm:gpu_bm.cpp
	g++ $(CFLAGS) -o $@ $< $(LDFLAGS)

cpu_bm:cpu_bm.cpp
	g++ $(CFLAGS) -o $@ $< $(LDFLAGS)

run:
	./cpu_bm

clean:
	rm *.o cpu_bm
