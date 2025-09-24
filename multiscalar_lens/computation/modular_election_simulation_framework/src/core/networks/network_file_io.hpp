#pragma once

#include "../../util/hdf5_util.hpp"

#include "../network.hpp"
#include "../agent.hpp"

#include "H5Cpp.h"


namespace BPsimulation::io {
	template<class Agent>
	void write_network_to_file(const SocialNetwork<Agent> *network, H5::H5File &file, const char* group_name="/network") {
		H5::Group group = file.createGroup(group_name);

		std::vector<std::vector<size_t>> neighbors;
		for (size_t node = 0; node < network->num_nodes(); ++node) {
			neighbors.push_back(network->neighbors(node));
		}
		util::hdf5io::H5WriteIrregular2DVector(group, neighbors, "neighbors");

		group.close();
	}
	template<class Agent>
	auto read_network_from_file(SocialNetwork<Agent> *network, H5::H5File &file, const char* group_name="/network") {
		H5::Group group = file.openGroup(group_name);

		std::vector<std::vector<size_t>> neighbors(0, std::vector<size_t>(0));
		util::hdf5io::H5ReadIrregular2DVector(group, neighbors, "neighbors");

		network->resize(neighbors.size());
		network->clear_connections();

		for (size_t node = 0; node < network->num_nodes(); ++node) {
			for (size_t neighbor : neighbors[node]) {
				network->add_connection_single_way(node, neighbor);
			}
		}

		group.close();
	}


	template<class Agent, class Agent2>
	void write_agent_states_to_file(const SocialNetwork<Agent> *network, const core::agent::AgentSerializerTemplate<Agent2> *serializer,
		H5::H5File &file, const char* group_name="/states")
	{
		static_assert(std::is_convertible<Agent, Agent2>::value, "Error: Agent class is not compatible with the one used by AgentSerializerTemplate in read_agent_states_from_file !");

		H5::Group group = file.createGroup(group_name);

		using variable_type = std::variant<bool, int, unsigned int, long, size_t, float, double>;
		using vector_variable_type = std::variant<
			std::vector<char>, std::vector<int>,    std::vector<unsigned int>,
			std::vector<long>, std::vector<size_t>, std::vector<float>,
			std::vector<double>>;

		std::vector<std::pair<std::string, int>> list_of_fields = serializer->list_of_fields();
		std::vector<vector_variable_type> write_vectors(list_of_fields.size());

		for (size_t ifield = 0; ifield < list_of_fields.size(); ++ifield) {
			switch (list_of_fields[ifield].second) {
			case 0:
				write_vectors[ifield] = std::vector<char>(        network->num_nodes());
				break;
			case 1:
				write_vectors[ifield] = std::vector<int>(         network->num_nodes());
				break;
			case 2:
				write_vectors[ifield] = std::vector<unsigned int>(network->num_nodes());
				break;
			case 3:
				write_vectors[ifield] = std::vector<long>(        network->num_nodes());
				break;
			case 4:
				write_vectors[ifield] = std::vector<size_t>(      network->num_nodes());
				break;
			case 5:
				write_vectors[ifield] = std::vector<float>(       network->num_nodes());
				break;
			case 6:
				write_vectors[ifield] = std::vector<double>(      network->num_nodes());
				break;
			}
		}

		for (size_t node = 0; node < network->num_nodes(); ++node) {
			std::vector<variable_type> values = serializer->write((Agent2&)(*network)[node]);

			for (size_t ifield = 0; ifield < list_of_fields.size(); ++ifield) {
				switch (list_of_fields[ifield].second) {
				case 0:
					group, std::get<0>(write_vectors[ifield])[node] = (char)std::get<0>(values[ifield]);
					break;
				case 1:
					group, std::get<1>(write_vectors[ifield])[node] = std::get<1>(values[ifield]);
					break;
				case 2:
					group, std::get<2>(write_vectors[ifield])[node] = std::get<2>(values[ifield]);
					break;
				case 3:
					group, std::get<3>(write_vectors[ifield])[node] = std::get<3>(values[ifield]);
					break;
				case 4:
					group, std::get<4>(write_vectors[ifield])[node] = std::get<4>(values[ifield]);
					break;
				case 5:
					group, std::get<5>(write_vectors[ifield])[node] = std::get<5>(values[ifield]);
					break;
				case 6:
					group, std::get<6>(write_vectors[ifield])[node] = std::get<6>(values[ifield]);
					break;
				}
			}
		}

		for (size_t ifield = 0; ifield < list_of_fields.size(); ++ifield) {
			switch (list_of_fields[ifield].second) {
			case 0:
				util::hdf5io::H5WriteVector<char>(        group, std::get<0>(write_vectors[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 1:
				util::hdf5io::H5WriteVector<int>(         group, std::get<1>(write_vectors[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 2:
				util::hdf5io::H5WriteVector<unsigned int>(group, std::get<2>(write_vectors[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 3:
				util::hdf5io::H5WriteVector<long>(        group, std::get<3>(write_vectors[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 4:
				util::hdf5io::H5WriteVector<size_t>(      group, std::get<4>(write_vectors[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 5:
				util::hdf5io::H5WriteVector<float>(       group, std::get<5>(write_vectors[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 6:
				util::hdf5io::H5WriteVector<double>(      group, std::get<6>(write_vectors[ifield]), list_of_fields[ifield].first.c_str());
				break;
			}
		}

		group.close();
	}

	template<class Agent, class Agent2>
	void read_agent_states_from_file(SocialNetwork<Agent> *network, const core::agent::AgentSerializerTemplate<Agent2> *serializer,
		H5::H5File &file, const char* group_name="/states")
	{
		static_assert(std::is_convertible<Agent2, Agent>::value, "Error: Agent class is not compatible with the one used by AgentSerializerTemplate in read_agent_states_from_file !");

		H5::Group group = file.openGroup(group_name);

		using variable_type = std::variant<bool, int, unsigned int, long, size_t, float, double>;
		using vector_variable_type = std::variant<
			std::vector<char>, std::vector<int>,    std::vector<unsigned int>,
			std::vector<long>, std::vector<size_t>, std::vector<float>,
			std::vector<double>>;

		std::vector<std::pair<std::string, int>> list_of_fields = serializer->list_of_fields();
		std::vector<vector_variable_type> read_vectors(list_of_fields.size());

		for (size_t ifield = 0; ifield < list_of_fields.size(); ++ifield) {
			switch (list_of_fields[ifield].second) {
			case 0:
				read_vectors[ifield] = std::vector<char>();
				util::hdf5io::H5ReadVector<char>(        group, std::get<0>(read_vectors[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 1:
				read_vectors[ifield] = std::vector<int>();
				util::hdf5io::H5ReadVector<int>(         group, std::get<1>(read_vectors[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 2:
				read_vectors[ifield] = std::vector<unsigned int>();
				util::hdf5io::H5ReadVector<unsigned int>(group, std::get<2>(read_vectors[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 3:
				read_vectors[ifield] = std::vector<long>();
				util::hdf5io::H5ReadVector<long>(        group, std::get<3>(read_vectors[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 4:
				read_vectors[ifield] = std::vector<size_t>();
				util::hdf5io::H5ReadVector<size_t>(      group, std::get<4>(read_vectors[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 5:
				read_vectors[ifield] = std::vector<float>();
				util::hdf5io::H5ReadVector<float>(       group, std::get<5>(read_vectors[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 6:
				read_vectors[ifield] = std::vector<double>();
				util::hdf5io::H5ReadVector<double>(      group, std::get<6>(read_vectors[ifield]), list_of_fields[ifield].first.c_str());
				break;
			}
		}

		std::vector<variable_type> values(list_of_fields.size());
		for (size_t node = 0; node < network->num_nodes(); ++node) {
			for (size_t ifield = 0; ifield < list_of_fields.size(); ++ifield) {
				switch (list_of_fields[ifield].second) {
				case 0:
					values[ifield] = (bool)std::get<0>(read_vectors[ifield])[node];
					break;
				case 1:
					values[ifield] = std::get<1>(read_vectors[ifield])[node];
					break;
				case 2:
					values[ifield] = std::get<2>(read_vectors[ifield])[node];
					break;
				case 3:
					values[ifield] = std::get<3>(read_vectors[ifield])[node];
					break;
				case 4:
					values[ifield] = std::get<4>(read_vectors[ifield])[node];
					break;
				case 5:
					values[ifield] = std::get<5>(read_vectors[ifield])[node];
					break;
				case 6:
					values[ifield] = std::get<6>(read_vectors[ifield])[node];
					break;
				}
			}

			serializer->read((Agent2&)(*network)[node], values);
		}

		group.close();
	}


	void write_counties_to_file(const std::vector<std::vector<size_t>> &counties, H5::H5File &file, const char* group_name="/counties") {
		H5::Group group = file.createGroup(group_name);
		util::hdf5io::H5WriteIrregular2DVector(group, counties, "counties");
		group.close();
	}

	void read_counties_from_file(std::vector<std::vector<size_t>> &counties, H5::H5File &file, const char* group_name="/counties") {
		counties.clear();

		H5::Group group = file.openGroup(group_name);
		util::hdf5io::H5ReadIrregular2DVector(group, counties, "counties");
		group.close();
	}


	void write_election_result_to_file(const core::election::ElectionResultTemplate *result, const core::election::ElectionResultSerializerTemplate *serializer,
		H5::H5File &file, const char* group_name="/election_result")
	{
		H5::Group group = file.createGroup(group_name);

		using variable_type = std::variant<bool, int, unsigned int, long, size_t, float, double>;

		std::vector<std::pair<std::string, int>> list_of_fields = serializer->list_of_fields();
		std::vector<variable_type>               values         = serializer->write(*result);

		for (size_t ifield = 0; ifield < list_of_fields.size(); ++ifield) {
			switch (list_of_fields[ifield].second) {
			case 0:
				util::hdf5io::H5WriteSingle<char>(        group, (char)std::get<0>(values[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 1:
				util::hdf5io::H5WriteSingle<int>(         group, std::get<1>(values[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 2:
				util::hdf5io::H5WriteSingle<unsigned int>(group, std::get<2>(values[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 3:
				util::hdf5io::H5WriteSingle<long>(        group, std::get<3>(values[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 4:
				util::hdf5io::H5WriteSingle<size_t>(      group, std::get<4>(values[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 5:
				util::hdf5io::H5WriteSingle<float>(       group, std::get<5>(values[ifield]), list_of_fields[ifield].first.c_str());
				break;
			case 6:
				util::hdf5io::H5WriteSingle<double>(      group, std::get<6>(values[ifield]), list_of_fields[ifield].first.c_str());
				break;
			}
		}

		group.close();
	}

	void write_election_results_to_file(const std::vector<core::election::ElectionResultTemplate*> &results, const core::election::ElectionResultSerializerTemplate *serializer,
		H5::H5File &file, const char* group_name="/election_results")
	{
		H5::Group group = file.createGroup(group_name);

		for (size_t i = 0; i < results.size(); ++i) {
			std::string this_election_group_name = std::string(group_name) + "/election_result_" + std::to_string(i);
			write_election_result_to_file(results[i], serializer, file, this_election_group_name.c_str());
		}

		group.close();
	}
}