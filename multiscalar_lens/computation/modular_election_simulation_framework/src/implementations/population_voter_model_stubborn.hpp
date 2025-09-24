#pragma once

#include <random>

#include "../core/agent_population/agent_population.hpp"
#include "voter_model_stubborn.hpp"

#include "../util/util.hpp"


namespace BPsimulation::implem {
	class AgentPopulationVoterstubborn : public core::agent::population::AgentPopulation<voter_stubborn> {
	public:
		double stubborn_equilibrium[2] = {0, 0};

		template<typename ...Args>
		void randomize(double mean_stubborn_equilibrium0, double mean_stubborn_equilibrium1, Args ...args) {
			std::normal_distribution<double> distribution(1.0, 0.5);
			stubborn_equilibrium[0] = std::max(0.d, distribution(util::get_random_generator())*mean_stubborn_equilibrium0);
			stubborn_equilibrium[1] = std::max(0.d, distribution(util::get_random_generator())*mean_stubborn_equilibrium1);

			AgentPopulation<voter_stubborn>::randomize(args...);
		}
		template<typename ...Args>
		void randomize(Args ...args) {
			randomize(0, 0, args...);
		}
	};

	class population_voter_stubborn_interaction_function : public core::agent::AgentInteractionFunctionTemplate<core::agent::population::AgentPopulation<voter_stubborn>> {
	public:
		size_t N_select;
		population_voter_stubborn_interaction_function(size_t N_select_) : N_select(N_select_) {}

		void operator()(core::agent::population::AgentPopulation<voter_stubborn> &agent,
			std::vector<std::pair<const core::agent::population::AgentPopulation<voter_stubborn>*, double>> neighbors) const
		{
			if (agent.population > 0) {
				std::vector<double> self_selected         = agent.random_select_self(N_select, {2, 3});
				std::vector<double> neighborhood_selected = agent.random_select(     N_select, neighbors, false);

				agent.integrate_population_variation({
					neighborhood_selected[0] + neighborhood_selected[2] - self_selected[0],
					neighborhood_selected[1] + neighborhood_selected[3] - self_selected[1],
					0,
					0});
			}
		}
	};

	class voter_stubborn_equilibirum_function : public core::agent::AgentWiseUpdateFunctionTemplate<AgentPopulationVoterstubborn> {
	public:
		double dt;

		voter_stubborn_equilibirum_function(double dt_=0.1) : dt(dt_) {}
		void operator()(AgentPopulationVoterstubborn &agent) const {
			double radicalization_flux0 = std::max(agent.proportions[2]-1, std::min(agent.proportions[0],
				dt*(agent.proportions[0]*agent.stubborn_equilibrium[0] - agent.proportions[2])));
			double radicalization_flux1 = std::max(agent.proportions[3]-1, std::min(agent.proportions[1],
				dt*(agent.proportions[1]*agent.stubborn_equilibrium[1] - agent.proportions[3])));

			agent.integrate_proportion_variation({
				-radicalization_flux0,
				-radicalization_flux1,
				 radicalization_flux0,
				 radicalization_flux1});
		}
	};

	class voter_stubborn_overtoon_effect : public core::election::ElectionRetroinfluenceTemplate<AgentPopulationVoterstubborn> {
	private:
		inline double overtoon_distance(double distance) const {
			double normalized_distance = 2*std::abs(distance) - 1;
			return std::pow(normalized_distance, exponent);
		}
	public:
		double dt, multiplier, exponent; 

		voter_stubborn_overtoon_effect(double dt_=0.1, double multiplier_=1, double exponent_=3) :
			dt(dt_), multiplier(multiplier_), exponent(exponent_) {}
		void operator()(AgentPopulationVoterstubborn& agent, const core::election::ElectionResultTemplate* election_result_) const {
			voter_majority_election_result *election_result = (voter_majority_election_result*)election_result_;

			double proportion = election_result->proportion;
			if (proportion > 0.5) {
				double overtoon_multiplier = dt*multiplier*overtoon_distance(proportion);

				double deradicalization_flux0 = overtoon_multiplier*agent.proportions[2];
				double unstubborn_flip_flux    = overtoon_multiplier*agent.proportions[0];
				double radicalization_flux1   = overtoon_multiplier*agent.proportions[1];

				agent.integrate_proportion_variation({
					deradicalization_flux0 - unstubborn_flip_flux,
					unstubborn_flip_flux    - radicalization_flux1,
					                       - deradicalization_flux0,
					radicalization_flux1});
			} else if (proportion < 0.5) {
				double overtoon_multiplier = dt*multiplier*overtoon_distance(1 - proportion);

				double deradicalization_flux1 = overtoon_multiplier*agent.proportions[3];
				double unstubborn_flip_flux    = overtoon_multiplier*agent.proportions[1];
				double radicalization_flux0   = overtoon_multiplier*agent.proportions[0];

				agent.integrate_proportion_variation({
					unstubborn_flip_flux   - radicalization_flux0,
					                      - deradicalization_flux1,
					radicalization_flux0,
					                      - deradicalization_flux1});
			}
		}
	};

	class voter_stubborn_frustration_effect : public core::election::ElectionRetroinfluenceTemplate<AgentPopulationVoterstubborn> {
	public:
		double dt, multiplier; 
		
		voter_stubborn_frustration_effect(double dt_=0.1, double multiplier_=1) : dt(dt_), multiplier(multiplier_) {}
		void operator()(AgentPopulationVoterstubborn& agent, const core::election::ElectionResultTemplate* election_result_) const {
			voter_majority_election_result *election_result = (voter_majority_election_result*)election_result_;

			if (election_result->result) {
				double radicalization_flux0 = dt*multiplier*agent.proportions[0];
				
				agent.integrate_proportion_variation({
					-radicalization_flux0,
					0,
					 radicalization_flux0,
					0});
			} else {
				double radicalization_flux1 = dt*multiplier*agent.proportions[1];

				agent.integrate_proportion_variation({
					0,
					-radicalization_flux1,
					0,
					 radicalization_flux1});
			}
		}
	};

	class AgentPopulationVoterstubbornSerializer : public core::agent::AgentSerializerTemplate<AgentPopulationVoterstubborn> {
	private:
		core::agent::population::AgentPopulationSerializer<voter_stubborn> partial_serializer;

	public:
		typedef std::variant<bool, int, unsigned int, long, size_t, float, double> variable_type;
		std::vector<std::pair<std::string, int>> list_of_fields() const {
			std::vector<std::pair<std::string, int>> list_of_fields_ = partial_serializer.list_of_fields();

			list_of_fields_.push_back({"stubborn_equilibrium_false", 6});
			list_of_fields_.push_back({"stubborn_equilibrium_true",  6});

			return list_of_fields_;
		}
		std::vector<variable_type> write(const AgentPopulationVoterstubborn &agent) const {
			std::vector<variable_type> values = partial_serializer.write(agent);

			values.push_back(agent.stubborn_equilibrium[0]);
			values.push_back(agent.stubborn_equilibrium[1]);

			return values;
		}
		void read(AgentPopulationVoterstubborn &agent, const std::vector<variable_type> &values) const {
			partial_serializer.read(agent, values);

			agent.stubborn_equilibrium[0] = std::get<double>(values[5]);
			agent.stubborn_equilibrium[1] = std::get<double>(values[6]);
		}
	};
}