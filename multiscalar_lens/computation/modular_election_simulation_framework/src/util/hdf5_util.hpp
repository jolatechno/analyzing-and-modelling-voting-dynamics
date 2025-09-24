#pragma once

#include "H5Cpp.h"


namespace util::hdf5io {
	H5::H5File open_truncate_if_needed(const char* filename) {
		H5::H5File file(filename, H5F_ACC_TRUNC);
		file.close();

		file.openFile(filename, H5F_ACC_RDWR);
		return file;
	}

	void H5flush_and_clean(H5::H5File &file, bool close_reopen=false) {
		H5Fflush(file.getId(), H5F_SCOPE_GLOBAL);
		H5garbage_collect();

		if (close_reopen) {
			auto filename = file.getFileName();
			file.close();
			file.openFile(filename, H5F_ACC_RDWR);
		}
	}

	const H5::DataType &H5DataType(bool X) {
		return H5::PredType::NATIVE_CHAR;
	}
	const H5::DataType &H5DataType(char X) {
		return H5::PredType::NATIVE_CHAR;
	}
	const H5::DataType &H5DataType(int X) {
		return H5::PredType::NATIVE_INT;
	}
	const H5::DataType &H5DataType(unsigned int X) {
		return H5::PredType::NATIVE_UINT;
	}
	const H5::DataType &H5DataType(long X) {
		return H5::PredType::NATIVE_LONG;
	}
	const H5::DataType &H5DataType(size_t X) {
		return H5::PredType::NATIVE_ULLONG;
	}
	const H5::DataType &H5DataType(float X) {
		return H5::PredType::NATIVE_FLOAT;
	}
	const H5::DataType &H5DataType(double X) {
		return H5::PredType::NATIVE_DOUBLE;
	}
	template<class Type>
	const H5::DataType &H5DataType(Type X) {
	static_assert(false, "Type not defined as serializable !");
		return H5::PredType::NATIVE_INT;
	}

	template<class Type>
	void H5WriteSingle(H5::Group &group, const Type &data, const char* data_name) {
		hsize_t dim[1] = { 1 };
	    H5::DataSpace dataspace = H5::DataSpace(1, dim);
	    H5::DataSet   dataset   = group.createDataSet(data_name, H5DataType(data), dataspace);

	    dataset.write(&data, H5DataType(data));
	    dataset.close();
	}
	template<class Type>
	Type H5ReadSingle(H5::Group &group, const char* data_name) {
		H5::DataSet   dataset   = group.openDataSet(data_name);
		H5::DataSpace dataspace = dataset.getSpace();

	    Type data;
	    dataset.read(&data, H5DataType(data), dataspace);

	    return data;
	}

	template<class Type>
	void H5WriteVector(H5::Group &group, const std::vector<Type> &data, const char* data_name) {
		hsize_t dim[1] = { data.size() };
	    H5::DataSpace dataspace = H5::DataSpace(1, dim);
	    H5::DataSet   dataset   = group.createDataSet(data_name, H5DataType(data[0]), dataspace);

	    dataset.write(data.data(), H5DataType(data[0]));
	    dataset.close();
	}
	template<class Type>
	void H5ReadVector(H5::Group &group, std::vector<Type> &data, const char* data_name) {
		H5::DataSet   dataset   = group.openDataSet(data_name);
		H5::DataSpace dataspace = dataset.getSpace();

	    hsize_t dims[1];
	    dataspace.getSimpleExtentDims(dims, NULL);

	    data.resize(dims[0]);
	    dataset.read(data.data(), H5DataType(data[0]), dataspace);
	    dataset.close();
	}

	template<class Type>
	void H5WriteIrregular2DVector(H5::Group &group, const std::vector<std::vector<Type>> &data, const char* data_name) {
		std::string begin_end_idx_name = std::string(data_name) + "_begin_end_idx";

		std::vector<size_t> begin_end_idx(data.size()+1, 0);
		for (size_t i = 0; i < data.size(); ++i) {
			begin_end_idx[i + 1] = begin_end_idx[i] + data[i].size();
		}
		H5WriteVector(group, begin_end_idx, begin_end_idx_name.c_str());

		std::vector<Type> flattend_data(begin_end_idx.back());
		for (size_t i = 0; i < data.size(); ++i) {
			for (size_t j = 0; j < data[i].size(); j++) {
				flattend_data[begin_end_idx[i] + j] = data[i][j];
			}
		}
		H5WriteVector(group, flattend_data, data_name);
	}
	template<class Type>
	void H5ReadIrregular2DVector(H5::Group &group, std::vector<std::vector<Type>> &data, const char* data_name) {
		std::string begin_end_idx_name = std::string(data_name) + "_begin_end_idx";

		std::vector<size_t> begin_end_idx(0);
		H5ReadVector(group, begin_end_idx, begin_end_idx_name.c_str());
		data.resize(begin_end_idx.size()-1);

		std::vector<Type> flattend_data(0);
		H5ReadVector(group, flattend_data, data_name);
		for (size_t i = 0; i < begin_end_idx.size()-1; ++i) {
			size_t this_size = begin_end_idx[i + 1] - begin_end_idx[i];
			data[i].resize(this_size);

			for (size_t j = 0; j < this_size; j++) {
				data[i][j] =  flattend_data[begin_end_idx[i] + j];
			}
		}
	}
}