#pragma once

#include "../agent.hpp"

#include "../../util/util.hpp"


namespace BPsimulation::core::agent::population {
	template<class Agent>
	class AgentPopulation : public AgentTemplate {
	private:
		std::vector<Agent> agent_types_;

	public:
		AgentPopulation() {
			Agent* mock_agent   = new Agent();
			       agent_types_ = mock_agent->list_of_possible_agents();
			delete mock_agent;

			proportions.resize(agent_types().size());
		}

		inline const std::vector<Agent> &agent_types() const {
			return agent_types_;
		}

		size_t population = 1;
		std::vector<double> proportions;

		void randomize(const std::vector<double> &mean_proportions, const std::vector<double> &proportions_var) {
			for (int i = 0; i < agent_types().size(); ++i) {
				std::normal_distribution<double> distribution(mean_proportions[i], proportions_var[i]);
				proportions[i] = std::max(1e-9d, distribution(util::get_random_generator()));
			}

			renormalize();
		}
		void randomize(const std::vector<double> &mean_proportions) {
			std::vector<double> proportions_var(agent_types().size());
			for (int ifield = 0; ifield < agent_types().size(); ++ifield) {
				proportions_var[ifield] = mean_proportions[ifield]/2;
			}
			randomize(mean_proportions, proportions_var);
		}
		void randomize() {
			std::vector<double> mean_proportions(proportions.size(), 1.d);
			randomize(mean_proportions);
		}
		template<typename... Args>
		void randomize(size_t pop_mean, size_t pop_var, Args ...args) {
			std::normal_distribution<double> distribution(pop_mean, pop_var);
			population = std::max((long)0, (long)distribution(util::get_random_generator()));

			randomize(args...);
		}

		void renormalize() {
			double normalization_factor = 0.0;
			for (double &proportion : proportions) {
				proportion            = std::max(0.d, std::min(1.d, proportion));
				normalization_factor += proportion;
			}
			for (double &proportion : proportions) {
				proportion /= normalization_factor;
			}
		}
		void integrate_proportion_variation(const std::vector<double> &agent_proportion_delta) {
			size_t num_fields = proportions.size();
			for (size_t ifield = 0; ifield < num_fields; ++ifield) {
				proportions[ifield] = proportions[ifield] + agent_proportion_delta[ifield];
			}

			renormalize();
		}
		void integrate_population_variation(const std::vector<double> &agent_populations_delta) {
			size_t num_fields = proportions.size();
			for (size_t ifield = 0; ifield < num_fields; ++ifield) {
				proportions[ifield] = (proportions[ifield]*population + agent_populations_delta[ifield])/population;
			}

			renormalize();
		}

		template<class Agent2>
		std::vector<double> random_select(
			size_t N_select,
			const std::vector<std::pair<const Agent2*, double>> neighbors,
			const bool include_self=false,
			const std::vector<size_t> &unselectable={}
		) const {
			static_assert(std::is_convertible<Agent2, AgentPopulation<Agent>>::value, "Error: Agent class is not compatible with the one used by AgentPopulationInteractionFunctionTemplate in random_select !");
			if (neighbors.empty() && !include_self) {
				Agent* mock_agent = new Agent();
				size_t num_fields = mock_agent->list_of_possible_agents().size();

				return std::vector<double>(num_fields, 0);
			}

			size_t num_fields;
			if (include_self) {
				num_fields = proportions.size();
			} else {
				num_fields = neighbors[0].first->proportions.size();
			}

			std::vector<char> is_selectable(num_fields, true);
			for (size_t unselectable_field : unselectable) {
				is_selectable[unselectable_field] = false;
			}

			std::vector<double> accumulated_proportions(num_fields, 0);
			double accumulated_population=0.d, normalization_factor=0.d;
			for (size_t ifield = 0; ifield < num_fields; ++ifield) {
				if (is_selectable[ifield]) {
					if (include_self) {
						accumulated_proportions[ifield] += proportions[ifield];
						accumulated_population          += population*proportions[ifield];
					}

					for (const auto [neighbor, weight] : neighbors) {
						accumulated_proportions[ifield] += weight*neighbor->proportions[ifield];
						accumulated_population          += weight*neighbor->population*neighbor->proportions[ifield];
					}

					normalization_factor += accumulated_proportions[ifield];
				}
			}

			std::vector<double> selected(num_fields, 0);
			if (normalization_factor == 0) {
				return selected;
			}

			long int to_select = std::min((long int)N_select, (long int)accumulated_population);
			if (to_select == 0) {
				return selected;
			}

			size_t last_idx = num_fields-1;
			while (!is_selectable[last_idx]) {
				--last_idx;
			}

			for (size_t ifield = 0; ifield <= last_idx; ++ifield) {
				if (ifield == last_idx) {
					selected[ifield] = to_select;
				} else if (is_selectable[ifield]) {
					double select_proportion = std::max(0.d, std::min(1.d,
						accumulated_proportions[ifield]/normalization_factor));

					std::binomial_distribution<long int> distribution(to_select, select_proportion);
					long int this_selected = distribution(util::get_random_generator());
					selected[ifield]       = (double)this_selected;

					to_select            -= this_selected;
					normalization_factor -= accumulated_proportions[ifield];
				}
			}

			return selected;
		}
		inline std::vector<double> random_select_self(size_t N_select, const std::vector<size_t> &unselectable={}) const {
			return random_select(N_select, std::vector<std::pair<const AgentPopulation<Agent>*, double>>{}, true, unselectable);
		}
	};

	template<class Agent>
	class PopulationElection : public election::ElectionTemplate<AgentPopulation<Agent>> {
	private:
		const election::ElectionTemplate<Agent> *base_election_func;
	public:
		PopulationElection(const election::ElectionTemplate<Agent> *electionfunc) : base_election_func(electionfunc) {}

		election::ElectionResultTemplate* get_neutral_election_result() const { 
			return base_election_func->get_neutral_election_result();
		};
		election::ElectionResultTemplate* operator()(const AgentPopulation<Agent>& agent) const {
			election::ElectionResultTemplate *result = get_neutral_election_result();
			for (int i = 0; i < agent.agent_types().size(); ++i) {
				election::ElectionResultTemplate *type_result = (*base_election_func)(agent.agent_types()[i]);

				int agent_type_population = int(agent.population*agent.proportions[i]);
				(*type_result) *= agent_type_population;

				(*result) += type_result;
			}

			return result;
		};
	};

	template<class Agent>
	class PopulationRenormalizeProportions : public AgentWiseUpdateFunctionTemplate<AgentPopulation<Agent>> {
	public:
		PopulationRenormalizeProportions() {}
		void operator()(AgentPopulation<Agent> &agent) const {
			agent.renormalize();
		}
	};

	template<class Agent>
	class AgentPopulationSerializer : public AgentSerializerTemplate<AgentPopulation<Agent>> {
	public:
		using variable_type = std::variant<bool, int, unsigned int, long, size_t, float, double>;
		
		std::vector<std::pair<std::string, int>> list_of_fields() const {
			Agent* mock_agent = new Agent();
			size_t num_fields = 1 + mock_agent->list_of_possible_agents().size();

			std::vector<std::pair<std::string, int>> list_of_fields_(num_fields);
			for (size_t ifield = 0; ifield < num_fields-1; ++ifield) {
				std::string field_name = "proportions_" + std::to_string(ifield);

				list_of_fields_[ifield] = {field_name, 6};
			}
			list_of_fields_[num_fields-1] = {"population", 4};

			return list_of_fields_;
		}
		std::vector<variable_type> write(const AgentPopulation<Agent> &agent) const {
			size_t num_fields = 1 + agent.proportions.size();
			std::vector<variable_type> values(num_fields);

			for (size_t ifield = 0; ifield < num_fields-1; ++ifield) {
				values[ifield] = agent.proportions[ifield];
			}
			values[num_fields-1] = agent.population;

			return values;
		}
		void read(AgentPopulation<Agent> &agent, const std::vector<variable_type> &values) const {
			size_t num_fields = 1 + agent.proportions.size();

			for (size_t ifield = 0; ifield < num_fields-1; ++ifield) {
				agent.proportions[ifield] = std::get<double>(values[ifield]);
			}
			agent.population = std::get<size_t>(values[num_fields-1]);
		}
	};
}