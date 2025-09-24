#pragma once

#include <numeric>
#include <cmath>
#include <numbers>
#include <math.h>

#include "math.hpp"


namespace util::map {
	template<typename Type>
	Type latLon_to_meter_distance(Type lat1, Type lon1, Type lat2, Type lon2) {
		static const Type R = 6371000; // Radius of the earth in m
		Type dLat = math::deg_to_rad(lat2-lat1);
		Type dLon = math::deg_to_rad(lon2-lon1); 
		Type a = std::sin(dLat/2)*std::sin(dLat/2) +
			std::cos(math::deg_to_rad(lat1))*std::cos(math::deg_to_rad(lat2))*std::sin(dLon/2)*std::sin(dLon/2); 
		Type c = 2*std::atan2(std::sqrt(a), std::sqrt(1-a)); 
		Type d = R*c; // Distance in m
		return d;
	}
}