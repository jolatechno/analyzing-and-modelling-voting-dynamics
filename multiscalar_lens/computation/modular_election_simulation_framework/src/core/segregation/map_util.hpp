#pragma once

#include "../../util/map_util.hpp"
#include "../../util/math.hpp"
#include "../../util/util.hpp"


namespace segregation::map::util {
	template<typename Type>
	std::vector<Type> get_distances_single(const std::vector<Type> &lat, const std::vector<Type> &lon, size_t idx) {
		std::vector<Type> distances(lat.size());

		for (size_t j = 0; j < lat.size(); ++j) {
			distances[j] = std::max((Type)1,
				::util::map::latLon_to_meter_distance(lat[idx], lon[idx], lat[j], lon[j]));
		}

		return distances;
	}

	template<typename Type>
	std::vector<std::vector<Type>> get_distances(const std::vector<Type> &lat, const std::vector<Type> &lon, const std::vector<size_t> &indexes) {
		std::vector<std::vector<Type>> distances(indexes.size(),  std::vector<Type>(lat.size()));

		for (size_t i = 0; i < indexes.size(); ++i) {
			distances[i] = get_distances_single(lat, lon, indexes[i]);
		}

		return distances;
	}

	template<typename Type>
	std::vector<std::vector<Type>> get_distances(const std::vector<Type> &lat, const std::vector<Type> &lon) {
		std::vector<std::vector<Type>> distances(lat.size(),  std::vector<Type>(lat.size()));

		for (size_t i = 0; i < lat.size(); ++i) {
			for (size_t j = 0; j < i; ++j) {
				distances[i][j] = std::max((Type)1,
					::util::map::latLon_to_meter_distance(lat[i], lon[i], lat[j], lon[j]));
				distances[j][i] = distances[i][j];
			}
			distances[i][i] = 0;
		}

		return distances;
	}
}