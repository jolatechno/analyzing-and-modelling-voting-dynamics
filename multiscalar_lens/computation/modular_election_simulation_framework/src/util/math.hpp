#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <numbers>


namespace util::math {
	template<typename Type, typename limitType1, typename limitType2>
	std::vector<Type> linspace(const limitType1 &min, const limitType2 &max, const size_t N) {
		std::vector<Type> vect(N);

		const Type Tmin  = (Type)min;
		const Type delta = ((Type)max - Tmin)/N;
		for (size_t i = 0; i < N; ++i) {
			vect[i] = Tmin + delta*i;
		}

		return vect;
	}

	template<typename Type, typename limitType1, typename limitType2>
	std::vector<Type> logspace(const limitType1 &min, const limitType2 &max, const size_t N) {
		std::vector<Type> vect(N);

		const Type log_min  = std::log((Type)min);
		const Type log_max  = std::log((Type)max);
		const Type delta = (log_max - log_min)/N;
		for (size_t i = 0; i < N; ++i) {
			vect[i] = std::exp(log_min + delta*i);
		}

		return vect;
	}


	template<typename Type>
	std::vector<size_t> get_sorted_indexes(const std::vector<Type> &Y) {
		std::vector<size_t> indexes(Y.size());
		std::iota(indexes.begin(), indexes.end(), 0);

		std::sort(indexes.begin(), indexes.end(), [&Y](size_t i, size_t j) {
			return Y[i] < Y[j];
		});

		return indexes;
	}


	template<typename Type>
	double get_KLdiv(const std::vector<Type> &P, const std::vector<Type> &Q) {
		double KLdiv = 0.d;
		static const double epsilon = 1e-18;

		for (size_t i = 0; i < P.size(); ++i) {
			if (P[i] > epsilon) {
				KLdiv += P[i]*std::log2(P[i]/std::max(epsilon, (double)Q[i]));
			}
		}

		return KLdiv;
	}

	template<typename Type>
	double get_KLdiv_single(const Type &P, const Type &Q) {
		return get_KLdiv({P, 1-P}, {Q, 1-Q});
	}


	template<typename Type>
	Type integral(const std::vector<Type> &Y, const std::vector<Type> &X) {
		Type result = 0;
		for (size_t i = 0; i < Y.size()-1; ++i) {
			result += (Y[i+1] + Y[i])/2*(X[i+1] - X[i]);
		}
		return result;
	}

	template<typename Type>
	Type integral(const std::vector<Type> &Y, const Type &dX) {
		return std::accumulate(Y.begin(), Y.end(), (Type)0)*dX;
	}

	template<typename Type, class Matrix>
	std::vector<Type> integrals(const Matrix &Y, const std::vector<Type> &X) {
		std::vector<Type> results(Y.size(), 0);
		for (size_t i = 0; i < Y.size(); ++i) {
			for (size_t j = 0; j < X.size()-1; ++j) {
				results[i] += (Y[i][j+1] + Y[i][j])/2*(X[j+1] - X[j]);
			}
		}
		return results;
	}

	template<typename Type, class Matrix>
	std::vector<Type> integrals(const Matrix &Y, const Type &dX) {
		std::vector<Type> results(Y.size(), 0);
		for (size_t i = 0; i < Y.size(); ++i) {
			for (size_t j = 0; j < Y[i].size(); ++j) {
				results[i] += Y[i][j];
			}
			results[i] *= dX;
		}
		return results;
	}

	template<typename Type>
	Type deg_to_rad(Type deg) {
		return deg*(std::numbers::pi_v<Type>/180);
	}
}