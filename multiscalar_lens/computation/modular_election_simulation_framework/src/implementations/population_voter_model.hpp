#pragma once

#include <random>

#include "../core/agent_population/agent_population.hpp"
#include "voter_model.hpp"

#include "../util/util.hpp"


namespace BPsimulation::implem {
	class population_voter_interaction_function : public core::agent::AgentInteractionFunctionTemplate<core::agent::population::AgentPopulation<voter>> {
	public:
		size_t N_select;
		population_voter_interaction_function(size_t N_select_) : N_select(N_select_) {}

		void operator()(core::agent::population::AgentPopulation<voter> &agent,
			std::vector<std::pair<const core::agent::population::AgentPopulation<voter>*, double>> neighbors) const
			{
			if (agent.population > 0) {
				std::vector<double> self_selected         = agent.random_select_self(N_select);
				std::vector<double> neighborhood_selected = agent.random_select(     N_select, neighbors, false);

				agent.integrate_population_variation({
					neighborhood_selected[0] - self_selected[0],
					neighborhood_selected[1] - self_selected[1]});
			}
		}
	};
}