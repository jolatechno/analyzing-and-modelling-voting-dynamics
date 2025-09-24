#pragma once

#include "../network.hpp"


namespace BPsimulation::random {
	template<class Agent>
	std::vector<std::vector<size_t>> random_graphAgnostic_partition_graph(SocialNetwork<Agent> *network, size_t n_partition) {
		std::vector<std::vector<size_t>> partition;

		std::vector<size_t> nodes = network->nodes();
		std::shuffle(nodes.begin(), nodes.end(), util::get_random_generator());

		for (size_t i = 0; i < n_partition; ++i) {
			size_t idx_begin = (network->num_nodes()* i  ) /n_partition;
			size_t idx_end   = (network->num_nodes()*(i+1))/n_partition;

			partition.push_back(std::vector<size_t>(nodes.begin() + idx_begin, nodes.begin() + idx_end));
		}

		return partition;
	}
}