#pragma once

#include "../../util/math.hpp"

#include "multiscalar.hpp"


namespace segregation::multiscalar::util {
	template<typename Type>
	std::vector<Type> get_total_distribution(const std::vector<std::vector<Type>> &vects) {
		std::vector<Type> total_distribution(vects.size());

		Type total = 0;
		#pragma omp parallel for
		for (size_t k = 0; k < vects.size(); ++k) {
			total_distribution[k] = std::accumulate(vects[k].begin(), vects[k].end(), 0);
			#pragma omp critical
			total                += total_distribution[k];
		}
		for (size_t k = 0; k < vects.size(); ++k) {
			total_distribution[k] /= total;
		}

		return total_distribution;
	}

	template<typename Type>
	std::vector<Type> get_populations(const std::vector<std::vector<Type>> &vects) {
		std::vector<Type> total_population(vects[0].size(), 0);

		Type total = 0;
		#pragma omp parallel for
		for (size_t i = 0; i < vects[0].size(); ++i) {
			for (size_t k = 0; k < vects.size(); ++k) {
				total_population[i] += vects[k][i];
			}
		}

		return total_population;
	}
	
	template<typename Type>
	std::vector<std::vector<Type>> get_accumulated_trajectory(const std::vector<std::vector<Type>> &vects, const std::vector<std::vector<size_t>> &indexes) {
		std::vector<std::vector<Type>> accumulated_trajectory(indexes.size(),  std::vector<Type>(indexes[0].size()));

		#pragma omp parallel for
		for (size_t i = 0; i < indexes.size(); ++i) {
			accumulated_trajectory[i][0] = 0;

			for (size_t j = 0; j < indexes[0].size(); ++j) {
				if (j > 0) {
					accumulated_trajectory[i][j] = accumulated_trajectory[i][j-1];
				}

				for (size_t k = 0; k < vects.size(); ++k) {
					accumulated_trajectory[i][j] += vects[k][indexes[i][j]];
				}
			}
		}

		return accumulated_trajectory;
	}

	template<typename Type>
	std::vector<std::vector<Type>> get_accumulated_trajectory_single(const std::vector<Type> &vect) {
		return get_accumulated_trajectory(std::vector<std::vector<Type>>({vect}));
	}

	template<typename Type=double>
	std::vector<Type> get_worst_Xvalues(const std::vector<std::vector<Type>> &Xvalues={}, size_t vect_size=0, bool is_reversed_order=false) {
		/* Get the X value trajectory,
		the "worst trajectory" is the one where we encounter the furthest 1st node,
		then the furhtest 2nd node, etc... */
		if (vect_size == 0) {
			vect_size = Xvalues.size();
		}
		std::vector<Type> worst_Xvalues(vect_size);

		if (Xvalues.empty()) {
			std::iota(worst_Xvalues.begin(), worst_Xvalues.end(), 0);
		} else {
			#pragma omp parallel for
			for (size_t i = 0; i < Xvalues.size(); ++i) {
				worst_Xvalues[i] = Xvalues[0][i];
				for (size_t j = 1; j < Xvalues[i].size(); ++j) {
					if (is_reversed_order) {
						worst_Xvalues[i] = std::min(worst_Xvalues[i], Xvalues[j][i]);
					} else {
						worst_Xvalues[i] = std::max(worst_Xvalues[i], Xvalues[j][i]);
					}
				}
			}
		}

		return worst_Xvalues;
	}

	template<typename Type>
	std::vector<Type> get_worst_population_trajectory(const std::vector<std::vector<Type>> &vects) {/* Get population of each node,
		the "worst trajectory" would be if we encounter the smallest to largest population: */
		std::vector<Type> populations = util::get_populations(vects);
		std::sort(populations.begin(), populations.end());

		std::partial_sum(populations.begin(), populations.end(), populations.begin());

		return populations;
	}

	template<typename Type>
	std::vector<Type> get_worst_KLdiv_trajectory(const std::vector<std::vector<Type>> &vects) {
		/* Get population of each node,
		the "worst trajectory" would be if we encounter the smallest to largest population: */
		std::vector<Type> populations = util::get_populations(vects);
		std::sort(populations.begin(), populations.end());
		const Type        total_pop   = std::accumulate(populations.begin(), populations.end(), (Type)0);

		/* Get the total distribution,
		sort it from smallest to biggest proportion,
		as the "worst trajectory" would be if we encounter the smallest minorities first */
		std::vector<Type> total_distribution = util::get_total_distribution(vects);
		std::sort(total_distribution.begin(), total_distribution.end());

		/* Get the population of each minority: */
		std::vector<Type> total_distribution_pop = total_distribution;
		for (Type &total_distribution_pop_ : total_distribution_pop) {
			total_distribution_pop_ *= total_pop;
		}


		std::vector<Type> KL_div_traj(vects[0].size());

		std::vector<Type> placeholder(vects.size()), accumulated_pop(vects.size(), 0);
		Type accumulated_total_pop = 0;
		int current_idx = 0;

		for (size_t i = 0; i < vects[0].size(); ++i) {
			accumulated_total_pop += populations[i];

			while (total_distribution_pop[current_idx] < populations[i] && current_idx < vects.size()-1) {
				accumulated_pop[current_idx] += total_distribution_pop[current_idx];
				populations[i]               -= total_distribution_pop[current_idx];
				total_distribution_pop[current_idx] = 0;
				++current_idx;
			}
			accumulated_pop[       current_idx] += populations[i];
			total_distribution_pop[current_idx] -= populations[i];

			for (size_t k = 0; k < vects.size(); ++k) {
				placeholder[k] = accumulated_pop[k]/accumulated_total_pop;
			}

			KL_div_traj[i] = ::util::math::get_KLdiv(placeholder, total_distribution);
		}


		return KL_div_traj;
	}
}