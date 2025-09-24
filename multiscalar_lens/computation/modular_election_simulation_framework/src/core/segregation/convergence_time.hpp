#pragma once

#include "../../util/math.hpp"

#include "multiscalar.hpp"
#include "multiscalar_util.hpp"


namespace segregation::convergence_time {
	template<typename Type>
	std::vector<std::vector<double>> get_KLdiv_trajectories_versus_trajectory_end(const std::vector<std::vector<std::vector<Type>>> &trajectories) {
		std::vector<std::vector<double>> KLdiv_trajectories(trajectories[0].size(), std::vector<double>(trajectories[0][0].size()));

		std::vector<Type> placeholder(trajectories.size()), distribution(trajectories.size());

		#pragma omp parallel for private(placeholder, distribution)
		for (size_t i = 0; i < trajectories[0].size(); ++i) {
			placeholder.resize( trajectories.size());
			distribution.resize(trajectories.size());

			for (size_t k = 0; k < trajectories.size(); ++k) {
				distribution[k] = trajectories[k][i].back();
			}

			for (size_t j = 0; j < trajectories[0][0].size(); ++j) {
				for (size_t k = 0; k < trajectories.size(); ++k) {
					placeholder[k] = trajectories[k][i][j];
				}
				KLdiv_trajectories[i][j] = ::util::math::get_KLdiv(placeholder, distribution);
			}
		}
		
		return KLdiv_trajectories;
	}

	using multiscalar::get_focal_distance_indexes;
	using multiscalar::get_focal_distances;

	using multiscalar::get_distortion_coefs;
	using multiscalar::get_distortion_coefs_from_KLdiv;
}