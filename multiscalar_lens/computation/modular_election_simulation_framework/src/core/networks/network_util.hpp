#pragma once

#include "../network.hpp"
#include "../agent.hpp"


namespace BPsimulation::random {
	template<class Agent, typename... Args>
	void inline network_randomize_agent_states_county(SocialNetwork<Agent> *network, const std::vector<size_t> &county, Args... args) {
		//#pragma omp parallel
		for (size_t node : county) {
			(*network)[node].randomize(args...);
		}
	}

	template<class Agent, typename... Args>
	void inline network_randomize_agent_states(SocialNetwork<Agent> *network, Args... args) {
		std::vector<size_t> node_list = network->nodes();
		network_randomize_agent_states_county(network, node_list, args...);
	}
}