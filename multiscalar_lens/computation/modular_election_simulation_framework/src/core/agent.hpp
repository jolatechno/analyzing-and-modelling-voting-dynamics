#pragma once

#include <variant>

#include "../util/util.hpp"


namespace BPsimulation::core::agent {
	class AgentTemplate {
	public:
		template<typename... Args>
		void randomize(Args... args) {}
	};

	template<class Agent>
	class AgentInteractionFunctionTemplate {
	protected:
		template<class Agent2>
		const Agent2* random_select(const std::vector<std::pair<const Agent2*, double>> neighbors) const {
			static_assert(std::is_convertible<Agent2, Agent>::value, "Error: Agent class is not compatible with the one used by AgentInteractionFunctionTemplate in random_select !");

			if (neighbors.empty()) {
				return NULL;
			}

			std::vector<double> probas(neighbors.size());

			probas[0] = neighbors[0].second;
			for (size_t i = 1; i < neighbors.size(); ++i) {
				probas[i] = probas[i-1] + neighbors[i].second;
			}

			std::uniform_real_distribution<double> distribution(0.d, probas.back());
			double rng_value = distribution(util::get_random_generator());

			size_t neighbor_idx = 0;
			while (rng_value > 0 && neighbor_idx <= neighbors.size()) {
				rng_value -= probas[neighbor_idx];
				++neighbor_idx;
			}
			--neighbor_idx;

			return neighbors[neighbor_idx].first;
		}

	public:
		virtual void operator()(Agent &agent, std::vector<std::pair<const Agent*, double>> neighbors) const {}
		virtual std::vector<Agent> list_of_possible_agents() const { return {}; }
	};

	template<class Agent>
	class AgentWiseUpdateFunctionTemplate {
	public:
		virtual void operator()(Agent &agent) const {}
	};

	template<class Agent>
	class AgentSerializerTemplate {
	protected:
		int type_id(bool          X) { return 0; }
		int type_id(int           X) { return 1; }
		int type_id(unsigned int  X) { return 2; }
		int type_id(long          X) { return 3; }
		int type_id(size_t        X) { return 4; }
		int type_id(float         X) { return 5; }
		int type_id(double        X) { return 6; }
		template<class Type>
		int type_id(Type          X) {
			static_assert(false, "Type not defined as serializable !");
			return -1;
		}

	public:
		typedef std::variant<bool, int, unsigned int, long, size_t, float, double> variable_type;

		virtual std::vector<std::pair<std::string, int>> list_of_fields() const { return {}; }
		virtual std::vector<variable_type> write(const Agent &agent) const { return {}; }
		virtual void read(Agent &agent, const std::vector<variable_type> &values) const {}
	};
}