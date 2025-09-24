all: test

par: test-par

all+par: all par

test:
	g++ -std=c++20 -O3 test.cpp -o test.out

test-par:
	g++ -std=c++20 -fopenmp -O3 test.cpp -o test-par.out