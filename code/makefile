all: tsp_sequential tsp_static tsp_dynamic tsp_fastflow clean

utils.o: src/utils.cpp src/genetic.h
	g++ -c -std=c++20 -O3 src/utils.cpp 
	
genetic.o: src/genetic.cpp src/genetic.h
	g++ -c -std=c++20 -O3 src/genetic.cpp 

utimer.o: src/utimer.cpp src/utimer.hpp
	g++ -c -std=c++20 -O3 src/utimer.cpp

tsp_sequential.o: tsp_sequential.cpp
	g++ -c -std=c++20 -O3 tsp_sequential.cpp 

tsp_static.o: tsp_static.cpp src/genetic.h
	g++ -c -std=c++20 -O3 tsp_static.cpp

tsp_dynamic.o: tsp_dynamic.cpp src/genetic.h
	g++ -c -std=c++20 -O3 tsp_dynamic.cpp

tsp_fastflow.o: tsp_fastflow.cpp src/genetic.h
	g++ -c -O3 -I/usr/local/include -L/usr/local/libs -lfastflow -std=c++20 -pthread -D NO_DEFAULT_MAPPING tsp_fastflow.cpp 

tsp_sequential: utimer.o tsp_sequential.o genetic.o utils.o
	g++ utimer.o tsp_sequential.o genetic.o utils.o -o tsp_sequential

tsp_static: utimer.o tsp_static.o genetic.o utils.o
	g++ utimer.o tsp_static.o genetic.o utils.o -o tsp_static -pthread

tsp_dynamic: utimer.o tsp_dynamic.o genetic.o utils.o
	g++ utimer.o tsp_dynamic.o genetic.o utils.o -o tsp_dynamic -pthread

tsp_fastflow: utimer.o tsp_fastflow.o genetic.o utils.o
	g++ utimer.o tsp_fastflow.o genetic.o utils.o -o tsp_fastflow -pthread

clean:
	rm -f *.o 

