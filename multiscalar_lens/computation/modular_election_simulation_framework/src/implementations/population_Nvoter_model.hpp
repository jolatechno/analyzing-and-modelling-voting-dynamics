#pragma once

#include <random>

#include "../core/agent_population/agent_population.hpp"
#include "Nvoter_model.hpp"

#include "../util/util.hpp"


namespace BPsimulation::implem {
	template<int N_candidates>
	class population_Nvoter_interaction_function : public core::agent::AgentInteractionFunctionTemplate<core::agent::population::AgentPopulation<Nvoter<N_candidates>>> {
	public:
		size_t N_select;
		population_Nvoter_interaction_function(size_t N_select_) : N_select(N_select_) {}

		void operator()(core::agent::population::AgentPopulation<Nvoter<N_candidates>> &agent,
			std::vector<std::pair<const core::agent::population::AgentPopulation<Nvoter<N_candidates>>*, double>> neighbors) const
		{
			if (agent.population > 0) {
				std::vector<double> self_selected         = agent.random_select_self(N_select);
				std::vector<double> neighborhood_selected = agent.random_select(     N_select, neighbors, false);

				std::vector<double> population_delta(N_candidates, 0);
				for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
					population_delta[icandidate] = neighborhood_selected[icandidate] - self_selected[icandidate];
				}

				agent.integrate_population_variation(population_delta);
			}
		}
	};
}