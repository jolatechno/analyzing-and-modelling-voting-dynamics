#pragma once

#include "agent_population.hpp"

#include "../agent.hpp"

#include "../../util/util.hpp"


namespace BPsimulation::core::agent::population::util {
	template<class Agent, class Agent2=AgentPopulation<Agent>>
	std::vector<std::vector<double>> get_vote_proportions(const SocialNetwork<Agent2> *network) {
		const size_t n_agent_type = (*network)[0].agent_types().size();

		std::vector<std::vector<double>> votes(n_agent_type, std::vector<double>(network->num_nodes(), 0));

		#pragma omp parallel
		for (size_t i = 0; i < network->num_nodes(); ++i) {
			size_t population = (*network)[i].population;
			for (size_t j = 0; j < n_agent_type; ++j) {
				double proportions = (*network)[i].proportions[j];
				votes[j][i] = population*proportions;
			}
		}

		return votes;
	}
}