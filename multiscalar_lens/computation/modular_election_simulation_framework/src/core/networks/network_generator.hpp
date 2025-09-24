#pragma once

#include <random>
#include <algorithm>

#include "../network.hpp"

#include "../../util/util.hpp"
#include "../../util/math.hpp"


namespace BPsimulation::random {
	template<class Agent>
	void preferential_attachment(SocialNetwork<Agent> *network, int n_attachment) {
		std::vector<size_t> degrees = network->degrees();
		size_t total_degree = std::accumulate(degrees.begin(), degrees.end(), 0);

		auto nodes = network->nodes();
		std::shuffle(nodes.begin(), nodes.end(), util::get_random_generator());

		if (total_degree == 0) {
			size_t i = nodes[network->num_nodes()-2];
			size_t j = nodes[network->num_nodes()-1];

			network->add_connection(i, j);

			degrees[i] += 1;
			degrees[j] += 1;
			total_degree = 2;
		}

		for (int i = 0; i < n_attachment; ++i) {
			for (size_t node : nodes) {
				size_t node_degree_save = degrees[node];
				total_degree -= node_degree_save;
				degrees[node] = 0;

				std::uniform_int_distribution<size_t> distribution(0, total_degree-1);
				size_t rdm = distribution(util::get_random_generator());

				for (size_t j = 0;; ++j) {
					if (rdm <= degrees[nodes[j]]) {
						degrees[nodes[j]] += 1;
						network->add_connection(node, nodes[j]);

						break;
					}
					rdm -= degrees[nodes[j]];
				}
				degrees[node]  = node_degree_save + 1;
				total_degree  += node_degree_save + 2;
			}
		}
	}

	template<class Agent, class Type>
	void closest_neighbor_limited_attachment(SocialNetwork<Agent> *network, const std::vector<std::vector<Type>> &distances,
		const int n_attachment, const int n_attachment_max=0)
	{
		const int n_attachment_max_ = std::max(n_attachment_max, n_attachment);

		std::vector<size_t>              idxs(          network->num_nodes(), 0);
		std::vector<std::vector<size_t>> sorted_indexes(network->num_nodes());
		for (size_t node = 0; node < network->num_nodes(); ++node) {
			sorted_indexes[node] = util::math::get_sorted_indexes(distances[node]);
		}

		auto nodes = network->nodes();
		std::shuffle(nodes.begin(), nodes.end(), util::get_random_generator());

		for (int i = 0; i < n_attachment; ++i) {
			for (size_t node : nodes) {
				if (network->neighbors(node).size() < n_attachment) {
					while (idxs[node] < network->num_nodes() && (
						   node == sorted_indexes[node][idxs[node]]                       ||
						   network->are_neighbors(node, sorted_indexes[node][idxs[node]]) ||
						   network->neighbors(sorted_indexes[node][idxs[node]]).size() >= n_attachment_max_))
					{
						++idxs[node];
					}

					if (idxs[node] < network->num_nodes()) {
						network->add_connection(node, sorted_indexes[node][idxs[node]]);
						++idxs[node];
					}
				}
			}
		}
	}
}