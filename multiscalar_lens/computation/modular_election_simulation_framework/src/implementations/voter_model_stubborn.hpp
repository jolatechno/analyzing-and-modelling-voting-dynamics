#pragma once

#include <random>

#include "voter_model.hpp"

#include "../util/util.hpp"


namespace BPsimulation::implem {
	class voter_stubborn : public voter {
	private:
		voter_stubborn(bool candidate_, bool stubborn_) : stubborn(stubborn_) {
			candidate = candidate_;
		}
	public:
		voter_stubborn() {}

		bool stubborn=false;
		
		void randomize(float p=0.5, float p_stubborn=0) {
			std::uniform_real_distribution<float> distribution(0.0, 1.0);
			
			candidate = distribution(util::get_random_generator()) < p;
			stubborn   = distribution(util::get_random_generator()) < p_stubborn;
		}

		std::vector<voter_stubborn> list_of_possible_agents() {
			std::vector<voter_stubborn> possible_agents;
			possible_agents.push_back(voter_stubborn(false, false));
			possible_agents.push_back(voter_stubborn(true,  false));
			possible_agents.push_back(voter_stubborn(false, true));
			possible_agents.push_back(voter_stubborn(true,  true));
			return possible_agents;
		}
	};

	class voter_stubborn_interaction_function : public core::agent::AgentInteractionFunctionTemplate<voter_stubborn> {
	public:
		void operator()(voter_stubborn &agent, std::vector<std::pair<const voter_stubborn*, double>> neighbors) const {
			if (!agent.stubborn) {
				const voter_stubborn* neighbor = random_select(neighbors);
				agent.candidate = neighbor->candidate;
			}
		}
	};

	class voter_stubborness_result : public core::election::ElectionResultTemplate {
	public:
		size_t candidate0_notstubborn=0, candidate1_notstubborn=0, candidate0_stubborn=0, candidate1_stubborn=0;
		float proportions[4] = {0, 0, 0, 0};

		voter_stubborness_result() {};
		core::election::ElectionResultTemplate& operator+=(const core::election::ElectionResultTemplate* other_) {
			voter_stubborness_result *other = (voter_stubborness_result*)other_;

			candidate0_notstubborn += other->candidate0_notstubborn;
			candidate1_notstubborn += other->candidate1_notstubborn;
			candidate0_stubborn    += other->candidate0_stubborn;
			candidate1_stubborn    += other->candidate1_stubborn;
			return *this;
		}
		ElectionResultTemplate& operator*=(size_t N) {
			candidate0_notstubborn *= N;
			candidate1_notstubborn *= N;
			candidate0_stubborn    *= N;
			candidate1_stubborn    *= N;
			return *this;
		};
		void post_process() {
			size_t total = candidate0_notstubborn + candidate1_notstubborn + candidate0_stubborn + candidate1_stubborn;
			proportions[0] = ((float)candidate0_notstubborn)/((float)total);
			proportions[1] = ((float)candidate1_notstubborn)/((float)total);
			proportions[2] = ((float)candidate0_stubborn   )/((float)total);
			proportions[3] = ((float)candidate1_stubborn   )/((float)total);
		}

	};

	class voter_stubborness_election : public core::election::ElectionTemplate<voter_stubborn> {
	public:
		voter_stubborness_result* get_neutral_election_result() const {
			return new voter_stubborness_result();
		}
		voter_stubborness_result* operator()(const voter_stubborn& agent) const {
			voter_stubborness_result *result = new voter_stubborness_result();

			result->candidate0_notstubborn = (!agent.candidate) && (!agent.stubborn);
			result->candidate1_notstubborn =   agent.candidate  && (!agent.stubborn);
			result->candidate0_stubborn    = (!agent.candidate) &&   agent.stubborn;
			result->candidate1_stubborn    =   agent.candidate  &&   agent.stubborn;

			return result;
		}
	};

	class VoterstubbornSerializer : public core::agent::AgentSerializerTemplate<voter_stubborn> {
	public:
		std::vector<std::pair<std::string, int>> list_of_fields() const {
			return {
				{"candidate", 0},
				{"stubborn",   0}
			};
		}
		std::vector<variable_type> write(const voter_stubborn &agent) const {
			std::vector<variable_type> values(2);

			values[0] = agent.candidate;
			values[1] = agent.stubborn;

			return values;
		}
		void read(voter_stubborn &agent, const std::vector<variable_type> &values) const {
			agent.candidate = std::get<bool>(values[0]);
			agent.stubborn   = std::get<bool>(values[1]);
		}
	};
}