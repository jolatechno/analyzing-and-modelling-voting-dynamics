#pragma once

#include <random>
#include <ostream>
#include <omp.h>


int max_print = 20;
template<typename objClass>
std::ostream &operator<<(std::ostream &os, const std::vector<objClass> &obj) {
	for (int i = 0; i < obj.size(); ++i) {
		os << obj[i];
		if (i == max_print) {
			os << "...";
			break;
		}
		if (i != obj.size()-1) {
			os << ", ";
		}
	}
    return os;
}


namespace util {
	namespace parallel {
		const int num_threads = []() {
		#if defined(_OPENMP)
			omp_set_max_active_levels(1);

			int num_threads_;

			#pragma omp parallel
			#pragma omp single
			num_threads_ = omp_get_num_threads();

			return num_threads_;
		#else
			return 1;
		#endif
		}();
	}

	namespace {
		std::vector<std::mt19937> random_generators = []() {
			std::vector<std::mt19937> random_generators_;

			std::random_device rand_dev;
			for (int i = 0; i < parallel::num_threads; ++i) {
				random_generators_.emplace_back(rand_dev());
			}

			return random_generators_;
		}();
	}
	
	void set_generator_seed(size_t seed) {
		for (int i = 0; i < parallel::num_threads; ++i) {
			random_generators[i].seed(seed);
		}
	}

	inline std::mt19937& get_random_generator() {
	#if defined(_OPENMP)
		return random_generators[omp_get_thread_num()];
	#else
		return random_generators[0];
	#endif
	}

	std::string get_first_cmd_arg(int argc, char *argv[]) {
		std::string cmd_arg = "";

		if (argc < 2) {
			return cmd_arg;
		}

		return argv[1];
	}
}