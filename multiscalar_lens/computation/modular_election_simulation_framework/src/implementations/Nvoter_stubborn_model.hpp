#pragma once

#include <random>
#include <algorithm>

#include "../core/election.hpp"
#include "../core/agent.hpp"

#include "../util/util.hpp"

#include "Nvoter_model.hpp"


namespace BPsimulation::implem {
	template<int N_candidates>
	class Nvoter_stubborn : public Nvoter<N_candidates> {
	private:
		Nvoter_stubborn(int candidate_, bool stubborn_) : stubborn(stubborn_) {
			Nvoter<N_candidates>::candidate = candidate_;
		}
	public:
		Nvoter_stubborn() {}

		bool stubborn=false;

		template<typename ...Args>
		void randomize(float p_stubborn=0, Args... args) {
			std::uniform_real_distribution<float> distribution(0.0, 1.0);
			stubborn = distribution(util::get_random_generator()) < p_stubborn;

			Nvoter<N_candidates>::randomize(args...);
		}

		std::vector<Nvoter_stubborn<N_candidates>> list_of_possible_agents() {
			std::vector<Nvoter_stubborn<N_candidates>> possible_agents;

			for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
				possible_agents.push_back(Nvoter_stubborn<N_candidates>(icandidate, false));
			}
			for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
				possible_agents.push_back(Nvoter_stubborn<N_candidates>(icandidate, true));
			}

			return possible_agents;
		}
	};

	template<int N_candidates>
	class Nvoter_stubborn_interaction_function : public core::agent::AgentInteractionFunctionTemplate<Nvoter_stubborn<N_candidates>> {
	public:
		void operator()(Nvoter_stubborn<N_candidates> &agent, std::vector<std::pair<const Nvoter_stubborn<N_candidates>*, double>> neighbors) const {
			if (!agent.stubborn) {
				const Nvoter_stubborn<N_candidates>* neighbor = random_select(neighbors);
				agent.candidate = neighbor->candidate;
			}
		}
	};

	template<int N_candidates>
	class NVoterstubbornSerializer : public core::agent::AgentSerializerTemplate<Nvoter_stubborn<N_candidates>> {
	public:
		typedef std::variant<bool, int, unsigned int, long, size_t, float, double> variable_type;

		std::vector<std::pair<std::string, int>> list_of_fields() const {
			return {
				{"candidate", 1},
				{"stubborn",   0}
			};
		}
		std::vector<variable_type> write(const Nvoter_stubborn<N_candidates> &agent) const {
			std::vector<variable_type> values(1);
			
			values[0] = agent.candidate;
			values[1] = agent.stubborn;

			return values;
		}
		void read(Nvoter_stubborn<N_candidates> &agent, const std::vector<variable_type> &values) const {
			agent.candidate = std::get<int >(values[0]);
			agent.stubborn   = std::get<bool>(values[1]);
		}
	};
}