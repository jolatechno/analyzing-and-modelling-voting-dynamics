#pragma once

#include <random>

#include "../core/election.hpp"
#include "../core/agent.hpp"

#include "../util/util.hpp"


namespace BPsimulation::implem {
	class voter : core::agent::AgentTemplate {
	private:
		voter(bool candidate_) : candidate(candidate_) {}
	public:
		voter() {}

		bool candidate = false;
		
		void randomize(float p=0.5) {
			std::uniform_real_distribution<float> distribution(0.0, 1.0);
			
			candidate = distribution(util::get_random_generator()) < p;
		}

		std::vector<voter> list_of_possible_agents() {
			std::vector<voter> possible_agents;
			possible_agents.push_back(voter(false));
			possible_agents.push_back(voter(true));
			return possible_agents;
		}
	};

	class voter_majority_election_result : public core::election::ElectionResultTemplate {
	public:
		size_t vote_True=0, vote_False=0;
		bool result;
		float proportion;

		voter_majority_election_result() {};
		core::election::ElectionResultTemplate& operator+=(const core::election::ElectionResultTemplate* other_) {
			voter_majority_election_result *other = (voter_majority_election_result*)other_;

			vote_True  += other->vote_True;
			vote_False += other->vote_False;
			return *this;
		}
		core::election::ElectionResultTemplate& operator*=(size_t N) {
			vote_True  *= N;
			vote_False *= N;
			return *this;
		};
		void post_process() {
			proportion = ((float)vote_True)/((float)(vote_True + vote_False));
			result     = proportion > 0.5;
		}

	};

	template<class Agent>
	class voter_majority_election : public core::election::ElectionTemplate<Agent> {
	public:
		voter_majority_election_result* get_neutral_election_result() const {
			return new voter_majority_election_result();
		}
		voter_majority_election_result* operator()(const Agent& agent) const {
			voter_majority_election_result *result = new voter_majority_election_result();

			result->vote_True  =  agent.candidate;
			result->vote_False = !agent.candidate;

			return result;
		}
	};

	class voter_interaction_function : public core::agent::AgentInteractionFunctionTemplate<voter> {
	public:
		void operator()(voter &agent, std::vector<std::pair<const voter*, double>> neighbors) const {
			const voter* neighbor = random_select(neighbors);
			agent.candidate = neighbor->candidate;
		}
	};

	class VoterMajorityElectionSerializer : public core::election::ElectionResultSerializerTemplate {
		std::vector<std::pair<std::string, int>> list_of_fields() const {
			return {
				{"result",      0},
				{"proportion",  5},
				{"vote_false",  4},
				{"vote_true",   4}
			};
		}
		std::vector<variable_type> write(const core::election::ElectionResultTemplate &result_) const {
			voter_majority_election_result &result = (voter_majority_election_result&)result_;

			std::vector<variable_type> values(4);

			values[0] = (bool)result.result;
			values[1] = result.proportion;
			values[2] = result.vote_False;
			values[3] = result.vote_True;

			return values;
		}
	};

	class VoterSerializer : public core::agent::AgentSerializerTemplate<voter> {
	public:
		std::vector<std::pair<std::string, int>> list_of_fields() const {
			return {
				{"candidate", 0}
			};
		}
		std::vector<variable_type> write(const voter &agent) const {
			std::vector<variable_type> values(1);
			
			values[0] = agent.candidate;

			return values;
		}
		void read(voter &agent, const std::vector<variable_type> &values) const {
			agent.candidate = std::get<bool>(values[0]);
		}
	};
}