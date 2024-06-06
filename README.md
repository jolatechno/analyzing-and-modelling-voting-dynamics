# Simulation setup for my PHD

The huge commad:

```bash
make all -j 8 && make par -j 8 && \
	./segregation-par.out 0paris && ./segregation-par.out 1petite-couronne &&  ./segregation-par.out 2region-paris && ./segregation-par.out 3france && \
	./simulation_1.out && \
	./simulation_2_from_data-par.out && \
	./simulation_3_convergence_time-par.out && \
	(cd ../postprocessing && \
		 ./postprocessing_segregation.py paris && ./postprocessing_segregation.py petite-couronne && ./postprocessing_segregation.py region-paris && ./postprocessing_segregation.py france && \
		./postprocessing_simulation_1.py && \
		./postprocessing_simulation_2_from_data.py && \
		./postprocessing_simulation_3_convergence_time.py)
```