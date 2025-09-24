#pragma once

#include <random>

#include "../core/agent_population/agent_population.hpp"
#include "voter_model_stubborn.hpp"

#include "../util/util.hpp"


namespace BPsimulation::implem {
	template<int N_candidates>
	class AgentPopulationNVoterstubborn : public core::agent::population::AgentPopulation<Nvoter_stubborn<N_candidates>> {
	public:
		std::vector<double> stubborn_equilibrium = std::vector<double>(N_candidates, 0.d);

		template<typename ...Args>
		void randomize(const std::vector<double> &mean_equilibirum, const std::vector<double> &equilibrium_var, Args ...args) {
			for (int i = 0; i < N_candidates; ++i) {
				std::normal_distribution<double> distribution(mean_equilibirum[i], equilibrium_var[i]);
				stubborn_equilibrium[i] = std::max(0.d, distribution(util::get_random_generator()));
			}

			core::agent::population::AgentPopulation<Nvoter_stubborn<N_candidates>>::randomize(args...);
		}
		template<typename ...Args>
		void randomize(const std::vector<double> &mean_equilibirum, Args ...args) {
			for (int i = 0; i < N_candidates; ++i) {
				std::normal_distribution<double> distribution(mean_equilibirum[i], mean_equilibirum[i]/2);
				stubborn_equilibrium[i] = std::max(0.d, distribution(util::get_random_generator()));
			}

			core::agent::population::AgentPopulation<Nvoter_stubborn<N_candidates>>::randomize(args...);
		}
		template<typename ...Args>
		void randomize(double mean_equilibirum, double equilibrium_var, Args ...args) {
			randomize(std::vector<double>(N_candidates, mean_equilibirum), std::vector<double>(N_candidates, equilibrium_var), args...);
		}
		template<typename ...Args>
		void randomize(double mean_equilibirum, Args ...args) {
			randomize(std::vector<double>(N_candidates, mean_equilibirum), args...);
		}
		template<typename ...Args>
		void randomize(Args ...args) {
			randomize(0, 0, args...);
		}
	};

	template<int N_candidates>
	class population_Nvoter_stubborn_interaction_function : public core::agent::AgentInteractionFunctionTemplate<core::agent::population::AgentPopulation<Nvoter_stubborn<N_candidates>>> {
	private:
		const std::vector<size_t> exclude_idx = []() {
			std::vector<size_t> exclude_idx_(N_candidates);
			std::iota(exclude_idx_.begin(), exclude_idx_.end(), N_candidates);

			return exclude_idx_;
		}();
	public:
		size_t N_select;
		population_Nvoter_stubborn_interaction_function(size_t N_select_) : N_select(N_select_) {}

		void operator()(core::agent::population::AgentPopulation<Nvoter_stubborn<N_candidates>> &agent,
			std::vector<std::pair<const core::agent::population::AgentPopulation<Nvoter_stubborn<N_candidates>>*, double>> neighbors) const
		{
			if (agent.population > 0) {
				std::vector<double> self_selected         = agent.random_select_self(N_select, exclude_idx);
				std::vector<double> neighborhood_selected = agent.random_select(     N_select, neighbors, false);

				std::vector<double> population_delta(2*N_candidates, 0);
				for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
					population_delta[icandidate] = neighborhood_selected[icandidate] + neighborhood_selected[N_candidates+icandidate] - self_selected[icandidate];
				}

				agent.integrate_population_variation(population_delta);
			}
		}
	};

	template<int N_candidates>
	class Nvoter_stubborn_equilibirum_function : public core::agent::AgentWiseUpdateFunctionTemplate<AgentPopulationNVoterstubborn<N_candidates>> {
	public:
		double dt;

		Nvoter_stubborn_equilibirum_function(double dt_=0.1) : dt(dt_) {}
		void operator()(AgentPopulationNVoterstubborn<N_candidates> &agent) const {
			std::vector<double> proportions_variations(2*N_candidates, 0.d);
			for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
				double radicalization_flux = std::max(agent.proportions[N_candidates+icandidate]-1, std::min(agent.proportions[icandidate],
					dt*(agent.proportions[icandidate]*agent.stubborn_equilibrium[icandidate] - agent.proportions[N_candidates+icandidate])));

				proportions_variations[             icandidate] = -radicalization_flux;
				proportions_variations[N_candidates+icandidate] =  radicalization_flux;
			}

			agent.integrate_proportion_variation(proportions_variations);
		}
	};

	template<int N_candidates>
	class Nvoter_stubborn_overtoon_effect : public core::election::ElectionRetroinfluenceTemplate<AgentPopulationNVoterstubborn<N_candidates>> {
	private:
		inline double overtoon_distance(double distance) const {
			double normalized_distance = std::abs(distance/((double)N_candidates));
			if (normalized_distance > radius) {
				normalized_distance = (radius - normalized_distance)/(1.d - radius);
				return -std::pow(-normalized_distance, exponent);
			} else {
				normalized_distance = (radius - normalized_distance)/radius;
				return std::pow(normalized_distance, exponent);
			}
		}
	public:	
		double dt, multiplier, radicalization_multiplier, radius, exponent;

		Nvoter_stubborn_overtoon_effect(double dt_=0.1, double multiplier_=1, double radicalization_multiplier_=1, double radius_=0.2, double exponent_=3) :
			dt(dt_), multiplier(multiplier_), radicalization_multiplier(radicalization_multiplier_), exponent(exponent_) {}
		void operator()(AgentPopulationNVoterstubborn<N_candidates>& agent, const core::election::ElectionResultTemplate* election_result_) const {
			Nvoter_majority_election_result<N_candidates> *election_result = (Nvoter_majority_election_result<N_candidates>*)election_result_;

			double election_mean_political_position=0.d, mean_political_position=0.d;
			for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
				election_mean_political_position += icandidate*election_result->proportions[icandidate];
				mean_political_position          += icandidate*          (agent.proportions[icandidate] + agent.proportions[N_candidates+icandidate]);
			}

			std::vector<double> proportions_variations(2*N_candidates, 0.d);
			for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
				double flux                = dt*multiplier*overtoon_distance(icandidate - election_mean_political_position);
				double radicalization_flux = radicalization_multiplier*flux;

				if (radicalization_flux > 0) {
					radicalization_flux *= agent.proportions[icandidate];

					proportions_variations[             icandidate] += -radicalization_flux;
					proportions_variations[N_candidates+icandidate] +=  radicalization_flux;
				} else {
					radicalization_flux *= agent.proportions[N_candidates+icandidate];

					proportions_variations[             icandidate] += -radicalization_flux;
					proportions_variations[N_candidates+icandidate] +=  radicalization_flux;
				}

				if (flux < 0) {
					flux *= agent.proportions[icandidate];
					if (mean_political_position > icandidate) {
						proportions_variations[icandidate]   +=  flux;
						proportions_variations[icandidate+1] += -flux;
					} else {
						proportions_variations[icandidate]   +=  flux;
						proportions_variations[icandidate-1] += -flux;
					}
				}
			}

			agent.integrate_proportion_variation(proportions_variations);
		}
	};

	template<int N_candidates>
	class Nvoter_stubborn_frustration_effect : public core::election::ElectionRetroinfluenceTemplate<AgentPopulationNVoterstubborn<N_candidates>> {
	public:
		double dt, multiplier; 
		
		Nvoter_stubborn_frustration_effect(double dt_=0.1, double multiplier_=1) : dt(dt_), multiplier(multiplier_) {}
		void operator()(AgentPopulationNVoterstubborn<N_candidates>& agent, const core::election::ElectionResultTemplate* election_result_) const {
			Nvoter_majority_election_result<N_candidates> *election_result = (Nvoter_majority_election_result<N_candidates>*)election_result_;

			std::vector<double> proportions_variations(2*N_candidates, 0.d);
			for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
				if (election_result->result != icandidate) {
					double radicalization_flux = dt*multiplier*agent.proportions[icandidate];

					proportions_variations[             icandidate] = -radicalization_flux;
					proportions_variations[N_candidates+icandidate] =  radicalization_flux;
				}
			}

			agent.integrate_proportion_variation(proportions_variations);
		}
	};

	template<int N_candidates>
	class AgentPopulationNVoterstubbornSerializer : public core::agent::AgentSerializerTemplate<AgentPopulationNVoterstubborn<N_candidates>> {
	private:
		core::agent::population::AgentPopulationSerializer<Nvoter_stubborn<N_candidates>> partial_serializer;

	public:
		typedef std::variant<bool, int, unsigned int, long, size_t, float, double> variable_type;
		std::vector<std::pair<std::string, int>> list_of_fields() const {
			std::vector<std::pair<std::string, int>> list_of_fields_ = partial_serializer.list_of_fields();

			for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
				std::string field_name = "stubborn_equilibrium_" + std::to_string(icandidate);
				list_of_fields_.push_back({field_name, 6});
			}

			return list_of_fields_;
		}
		std::vector<variable_type> write(const AgentPopulationNVoterstubborn<N_candidates> &agent) const {
			std::vector<variable_type> values = partial_serializer.write(agent);

			for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
				values.push_back(agent.stubborn_equilibrium[icandidate]);
			}

			return values;
		}
		void read(AgentPopulationNVoterstubborn<N_candidates> &agent, const std::vector<variable_type> &values) const {
			partial_serializer.read(agent, values);

			for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
				agent.stubborn_equilibrium[icandidate] = std::get<double>(values[1+2*N_candidates+icandidate]);
			}
		}
	};

	namespace util {
		void accumulate_stubborn_votes(std::vector<std::vector<double>> &votes) {
			int N_candidates = votes.size()/2;
			for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
				for (size_t i = 0; i < votes[0].size(); ++i) {
					votes[icandidate][i] += votes[N_candidates + icandidate][i];
				}
			}

			votes.resize(N_candidates);
		}
	}
}