library(readr)
library(readxl)

library(stringr)
library(tools)

library(rhdf5)
library(rjson)

# read config
read_config <- function(filename) {
	cmd_args <- commandArgs(trailingOnly=TRUE)
	if (length(cmd_args) == 0) {
		json_file <- fromJSON(file=filename)
		return(json_file)
	} else {
		json_file <- fromJSON(file=filename)
		return(json_file[[cmd_args[1]]])
	}
}

# read or compute then write function
read_file_or_compute <- function(funcRead, funcWrite, funcCompute) {
	return(tryCatch(
	{
		return(suppressMessages(suppressWarnings(funcRead())))
	},
	error=function(cond) {
		data <- funcCompute()
		suppressMessages(suppressWarnings(try(funcWrite(data))))
		return(data)
	}))
}

# read functions
read_map <- function(filename) {
	return(function() {
		data <- st_read(filename, quiet=TRUE)
		print(paste("Read from '", filename, "'", sep=""))
		return(data)
	})
}
read_vector_or_df_or_matrix <- function(filename) {
	return(function() {
		data <- h5read(filename, "data/df")
		print(paste("Read from '", filename, "'", sep=""))
		return(data)
	})
}
read_matrices <- function(filename) {
	return(function() {
		cols <- h5read(filename, "data/colnames")
		data <- FALSE

		if (length(cols) == 0) {
			stop("empty file")
		}

		for (col in cols) {
			traj <- h5read(filename, paste("data/", col, sep=""))

			# initiate dataframe
			if (col == cols[1]) {
				data <- data.frame(mat=rep(0, nrow(traj)))
				for (col_ in cols) {
					data[[col_]] <- matrix(0, nrow=nrow(traj), ncol=ncol(traj))
				}
			}

			# copy read data to dataframe
			data[[col]] <- traj
		}

		return(data)
	})
}

# write functions
write_map <- function(filename) {
	return(function(data) {
		print(paste("Writing to '", filename, "'...", sep=""))
		try(file.remove(filename))
		st_write(data, filename, quiet=TRUE)
	})
}
write_vector_or_df_or_matrix <- function(filename) {
	return(function(data) {
		print(paste("Writing to '", filename, "'...", sep=""))
		try(file.remove(filename))
		h5createFile(filename)
		h5createGroup(filename, "data")
		h5write(data, filename, "data/df")
	})
}
write_matrices <- function(filename) {
	return(function(data) {
		print(paste("Writing to '", filename, "'...", sep=""))
		try(file.remove(filename))
		h5createFile(filename)
		h5createGroup(filename, "data")

		colnames_base  <- c(colnames(data))
		colnames_write <- colnames_base[colnames_base != "mat"]

		h5write(colnames_write, filename, "data/colnames")
		for (col in colnames_write) {
			h5write(data[[col]], filename, paste("data/", col, sep=""))
		}
	})
}