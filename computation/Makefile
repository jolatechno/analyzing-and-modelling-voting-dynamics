all: test simulation_1 simulation_2_from_data simulation_3_convergence_time segregation

par: test-par segregation-par simulation_2_from_data-par

all+par: all par

test:
	$(MAKE) -C ../modular_election_simulation_framework test

test-par:
	$(MAKE) -C ../modular_election_simulation_framework test-par

simulation_1:
	h5c++ -O3 simulation_1.cpp -o simulation_1.out

simulation_2_from_data:
	h5c++ -ljsoncpp -O3 simulation_2_from_data.cpp -o simulation_2_from_data.out

simulation_2_from_data-par:
	h5c++ -ljsoncpp -fopenmp -O3 simulation_2_from_data.cpp -o simulation_2_from_data-par.out

simulation_3_convergence_time:
	h5c++ -ljsoncpp -O3 simulation_3_convergence_time.cpp -o simulation_3_convergence_time.out

segregation:
	h5c++ -ljsoncpp -O3 segregation.cpp -o segregation.out

segregation-par:
	h5c++ -ljsoncpp -fopenmp -O3 segregation.cpp -o segregation-par.out