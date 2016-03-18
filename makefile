CC=g++
CFLAGS=`pkg-config opencv --cflags`
LDFLAGS=`pkg-config opencv --libs`

all: cpu_bm

cpu_bm:cpu_bm.o
	g++ -o $@ $^ $(LDFLAGS)

cpu_bm.o: cpu_bm.cpp
	g++ $(CFLAGS) -c $<

run:
	./cpu_bm

clean:
	rm *.o cpu_bm
