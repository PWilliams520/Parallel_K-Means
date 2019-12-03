all: main.cpp
	g++ --std=c++11 -fopenmp main.cpp -lboost_filesystem -lboost_system -o par_kmeans

clean:
	rm par_kmeans