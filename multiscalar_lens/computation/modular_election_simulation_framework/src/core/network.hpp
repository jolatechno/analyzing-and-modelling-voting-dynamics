#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <random>

#include "election.hpp"
#include "agent.hpp"

#include "../util/util.hpp"


namespace BPsimulation {
	template<class Agent>
	class SocialNetwork {
	private:
		std::vector<Agent>               agent_vect, placeholder;
		std::vector<std::vector<size_t>> connection_matrix; 
		std::vector<std::vector<double>> weight_matrix; 


		template<class Agent2>
		std::vector<std::pair<const Agent2*, double>> get_neighbors(size_t node) const {
			static_assert(std::is_convertible<Agent, Agent2>::value, "Error: Agent class is not compatible with the one seeked in get_neighbors (private function of SocialNetwork) !");

			std::vector<std::pair<const Agent2*, double>> vec;

			std::vector<size_t> neighbor_list = neighbors(node);
			for (size_t neighbor_idx = 0; neighbor_idx < neighbor_list.size(); ++neighbor_idx) {
				size_t neighbor = neighbor_list[neighbor_idx];

				vec.push_back(std::pair<const Agent2*, double>{
					(const Agent2*)&(*this)[neighbor],
					weight_matrix[node][neighbor_idx]
				});
			}

			return vec;
		}

		std::pair<bool, size_t> get_neighbor_idx(size_t i, size_t j) const {
			std::vector<size_t> i_neighbors = neighbors(i);

			auto ptr = std::find(i_neighbors.begin(), i_neighbors.end(), j);
			if (ptr == i_neighbors.end()) {
				return {false, 0};
			} else {
				size_t idx = std::distance(i_neighbors.begin(), ptr);
				return {true, idx};
			}
		}
	public:
		SocialNetwork(size_t num_nodes=0) {
			resize(num_nodes);
		}

		inline size_t num_nodes() const {
			return agent_vect.size();
		}
		inline void resize(size_t num_nodes) {
			agent_vect.resize(       num_nodes);
			connection_matrix.resize(num_nodes);
			weight_matrix.resize(    num_nodes);
		}
		inline std::vector<size_t> nodes() const {
			std::vector<size_t> nodes(num_nodes());
			std::iota(nodes.begin(), nodes.end(), 0);
			return nodes;
		}
		inline Agent& operator[](size_t node) {
			return agent_vect[node];
		}
		inline const Agent& operator[](size_t node) const {
			return agent_vect[node];
		}

		inline const std::vector<size_t>& neighbors(size_t node) const {
			return connection_matrix[node];
		}

		inline double& get_connection_weight_ref(size_t i, size_t j) {
			auto [are_connected, idx] = get_neighbor_idx(i, j);

			if (!are_connected) {
				throw std::invalid_argument("in \"get_connection_weight_ref\", i and j aren't neighors, can't return reference");
			}
			return weight_matrix[i][idx];
		}
		inline const double get_connection_weight(size_t i, size_t j) const {
			auto [are_connected, idx] = get_neighbor_idx(i, j);
			if (are_connected) {
				return weight_matrix[i][idx];
			} else {
				return 0;
			}
		}
		inline void set_connection_weight_one_way(size_t i, size_t j, double weight) {
			auto [are_connected, idx] = get_neighbor_idx(i, j);
			if (are_connected) {
				weight_matrix[i][idx] = weight;
			} else {
				connection_matrix[i].push_back(j);
				weight_matrix[    i].push_back(weight);
			}
		}
		inline void set_connection_weight(size_t i, size_t j, double weight_ij, double weight_ji) {
			set_connection_weight_one_way(i, j, weight_ij);
			set_connection_weight_one_way(j, i, weight_ji);
		}
		inline void set_connection_weight(size_t i, size_t j, double weight) {
			set_connection_weight(i, j, weight, weight);
		}
		inline double increment_connection_weight_one_way(size_t i, size_t j, double weight) {
			auto [are_connected, idx] = get_neighbor_idx(i, j);
			if (are_connected) {
				weight_matrix[i][idx] += weight;
				return weight_matrix[i][idx];
			} else {
				connection_matrix[i].push_back(j);
				weight_matrix[    i].push_back(weight);
				return weight;
			}
		}
		inline std::pair<double, double> increment_connection_weight(size_t i, size_t j, double weight_ij, double weight_ji) {
			double updated_weight_ij = increment_connection_weight_one_way(i, j, weight_ij);
			double updated_weight_ji = increment_connection_weight_one_way(j, i, weight_ji);
			return {updated_weight_ij, updated_weight_ji};
		}
		inline std::pair<double, double> increment_connection_weight(size_t i, size_t j, double weight) {
			return increment_connection_weight(i, j, weight, weight);
		}

		inline bool are_neighbors(size_t i, size_t j) const {
			if (i == j) {
				return true;
			}
			auto [are_connected, idx] = get_neighbor_idx(i, j);
			return are_connected;
		}
		inline void add_connection_single_way(size_t i, size_t j, double weight=1.d) {
			if (!are_neighbors(i, j)) {
				connection_matrix[i].push_back(j);
				weight_matrix[    i].push_back(weight);
			}
		}
		inline void add_connection(size_t i, size_t j, double weight=1.d) {
			add_connection_single_way(i, j, weight);
			add_connection_single_way(j, i, weight);
		}
		inline void add_connection(size_t i, size_t j, double weight_ij, double weight_ji) {
			add_connection_single_way(i, j, weight_ij);
			add_connection_single_way(j, i, weight_ji);
		}

		inline void remove_connection_single_way(size_t i, size_t j) {
			auto [are_connected, idx] = get_neighbor_idx(i, j);
			if (are_connected) {
				connection_matrix[i].erase(connection_matrix[i].begin() + idx);
				weight_matrix[    i].erase(weight_matrix[    i].begin() + idx);
			}
		}
		inline void remove_connection(size_t i, size_t j) {
			remove_connection_single_way(i, j);
			remove_connection_single_way(j, i);
		}
		inline void clear_connections(size_t i) {
			connection_matrix[i].clear();
			weight_matrix[    i].clear();
		}
		inline void clear_connections() {
			for (size_t node = 0; node < num_nodes(); ++node) {
				clear_connections(node);
			}
		}
		inline void cleanup_connections(size_t i, double epsilon) {
			for (long long int idx = weight_matrix.size()-1; idx > 0; --idx) {
				if (std::abs(weight_matrix[i][idx]) <= epsilon) {
					connection_matrix[i].erase(connection_matrix[i].begin() + idx);
					weight_matrix[    i].erase(weight_matrix[    i].begin() + idx);
				}
			}
		}
		inline void cleanup_connections(double epsilon) {
			for (size_t node = 0; node < num_nodes(); ++node) {
				cleanup_connections(node, epsilon);
			}
		}

		inline size_t degree(size_t node) const {
			return neighbors(node).size();
		}
		inline std::vector<size_t> degrees() const {
			std::vector<size_t> degrees(num_nodes());
			for (size_t node = 0; node < num_nodes(); ++node) {
				degrees[node] = degree(node);
			}
			return degrees;
		}

		template<class Agent2>
		inline void interact_serial(const core::agent::AgentInteractionFunctionTemplate<Agent2> *interactionfunc) {
			static_assert(std::is_convertible<Agent, Agent2>::value, "Error: Agent class is not compatible with the one used by AgentInteractionFunctionTemplate in interact_serial !");

			std::vector<size_t> node_lists = nodes();
			std::shuffle(node_lists.begin(), node_lists.end(), util::get_random_generator());

			for (size_t node : node_lists) {
				(*interactionfunc)((Agent2&)(*this)[node], get_neighbors<Agent2>(node));
			}
		}

		template<class Agent2>
		inline void interact_parallel(const core::agent::AgentInteractionFunctionTemplate<Agent2> *interactionfunc) {
			static_assert(std::is_convertible<Agent, Agent2>::value, "Error: Agent class is not compatible with the one used by AgentInteractionFunctionTemplate in interact_parallel !");

			placeholder.resize(num_nodes());
			#pragma omp parallel for
			for (size_t node = 0; node < num_nodes(); ++node) {
				placeholder[node] = (*this)[node];
			}
			#pragma omp parallel for
			for (size_t node = 0; node < num_nodes(); ++node) {
				(*interactionfunc)((Agent2&)placeholder[node], get_neighbors<Agent2>(node));
			}
			#pragma omp parallel for
			for (size_t node = 0; node < num_nodes(); ++node) {
				(*this)[node] = placeholder[node];
			}
		}

		template<class Agent2>
		inline void interact(const core::agent::AgentInteractionFunctionTemplate<Agent2> *interactionfunc, bool parallel=false) {
			static_assert(std::is_convertible<Agent, Agent2>::value, "Error: Agent class is not compatible with the one used by AgentInteractionFunctionTemplate in interact !");
			if (parallel) {
				interact_parallel(interactionfunc);
			} else {
				interact_serial(  interactionfunc);
			}
		}

		template<class Agent2>
		core::election::ElectionResultTemplate* get_election_results(const std::vector<size_t> &county, const core::election::ElectionTemplate<Agent2> *electionfunc) const {
			static_assert(std::is_convertible<Agent, Agent2>::value, "Error: Agent class is not compatible with the one used by ElectionTemplate in ElectionResultTemplate !");

			auto *result = electionfunc->get_neutral_election_result();
			for (auto it = county.begin(); it != county.end(); ++it) {
				(*result) += (*electionfunc)((Agent2&)(*this)[*it]);
			}

			result->post_process();
			return result;
		}
		template<class Agent2>
		inline core::election::ElectionResultTemplate* get_election_results(const core::election::ElectionTemplate<Agent2> *electionfunc) const {
			std::vector<size_t> node_lists = nodes();
			return get_election_results(node_lists, electionfunc);
		}
		template<class Agent2>
		inline std::vector<core::election::ElectionResultTemplate*> get_election_results(const std::vector<std::vector<size_t>> &counties, const core::election::ElectionTemplate<Agent2> *electionfunc) const {
			static_assert(std::is_convertible<Agent, Agent2>::value, "Error: Agent class is not compatible with the one used by ElectionTemplate in get_election_results !");

			std::vector<core::election::ElectionResultTemplate*> results(counties.size());
			#pragma omp parallel for
			for (size_t i = 0; i < counties.size(); ++i) {
				results[i] = get_election_results(counties[i], electionfunc);
			}
			return results;
		}

		template<class Agent2>
		void inline update_agentwise(const core::agent::AgentWiseUpdateFunctionTemplate<Agent2> *updatefunc) {
			static_assert(std::is_convertible<Agent, Agent2>::value, "Error: Agent class is not compatible with the one used by AgentWiseUpdateFunctionTemplate in update_agentwise !");

			#pragma omp parallel for
			for (size_t node = 0; node < num_nodes(); ++node) {
				(*updatefunc)((Agent2&)(*this)[node]);
			}
		}

		template<class Agent2>
		void inline election_retroinfluence(const std::vector<size_t> &county, const core::election::ElectionResultTemplate *election_results, const core::election::ElectionRetroinfluenceTemplate<Agent2> *influencefunc) {
			static_assert(std::is_convertible<Agent, Agent2>::value, "Error: Agent class is not compatible with the one used by ElectionRetroinfluenceTemplate in election_retroinfluence !");

			#pragma omp parallel for
			for (size_t node : county) {
				(*influencefunc)((Agent2&)(*this)[node], election_results);
			}
		}
		template<class Agent2>
		void inline election_retroinfluence(const core::election::ElectionResultTemplate *election_results, const core::election::ElectionRetroinfluenceTemplate<Agent2> *influencefunc) {
			std::vector<size_t> node_lists = nodes();
			return election_retroinfluence(node_lists, election_results, influencefunc);
		}
		template<class Agent2>
		void inline election_retroinfluence(const std::vector<std::vector<size_t>> &counties, const std::vector<core::election::ElectionResultTemplate*> &election_results, const core::election::ElectionRetroinfluenceTemplate<Agent2> *influencefunc) {
			static_assert(std::is_convertible<Agent,Agent2>::value, "Error: Agent class is not compatible with the one used by ElectionRetroinfluenceTemplate in election_retroinfluence !");

			size_t max_2nd_loop_length = counties[0].size();
			for (size_t i = 1; i < counties.size(); ++i) {
				max_2nd_loop_length = std::max(max_2nd_loop_length, counties[1].size());
			}

			#pragma omp parallel for collapse(2)
			for (size_t i = 0; i < counties.size(); ++i) {
				for (size_t j = 0; j < max_2nd_loop_length; ++j) {
					if (j < counties[i].size()) {
						size_t node = counties[i][j];
						(*influencefunc)((Agent2&)(*this)[node], election_results[i]);
					}
	#if !defined(_OPENMP)
					else {
						break;
					}
	#endif
				}
			}
		}
	};
}